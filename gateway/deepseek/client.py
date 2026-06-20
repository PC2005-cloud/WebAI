"""_DeepSeekHTTPClient — HTTP 直连 DeepSeek 内部 API。

调用流程
--------
    1. POST /api/v0/chat_session/create  → 获取 session_id
    2. 浏览器 PoW 计算                    → 获取 x-ds-pow-response
    3. POST /api/v0/chat/completion       → SSE 响应
    4. SSE 流解析                         → 响应文本

为什么用同步库？
---------------
httpx 和 Playwright 都是同步库，选择它们是因为:
1. PoW 浏览器跑在独立线程中（sync_playwright），与 async 不兼容
2. httpx 同步 API 配合 asyncio.to_thread 比纯 async 方案更简单
3. 与 FastAPI async 不冲突——所有同步调用都委托到线程池

DeepSeek 内部 API 说明
-----------------------
以下端点随时可能变更（已验证于 2026-06）:
    POST /api/v0/chat_session/create
        body: {}  (空对象)
        response: {"data": {"biz_data": {"id": "uuid"}}}

    POST /api/v0/chat/completion
        body: {
            "chat_session_id": "...",
            "parent_message_id": null,
            "model_type": "default",        # "default" | "expert" | "vision"
            "prompt": "...",
            "thinking_enabled": true,
            "search_enabled": true,
            ...
        }
        headers: {
            "Authorization": "Bearer {token}",
            "X-Ds-Pow-Response": "{pow_value}"
        }
        response: SSE 流（非标准格式，见 sse.py 的说明）
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Callable, Optional
from urllib.parse import urljoin

import httpx

from core.exceptions import (
    BackendNotAvailableError,
    PowTimeoutError,
    SessionExpiredError,
)
from gateway.deepseek.pow import PowSolver
from gateway.deepseek.session import load_token
from gateway.deepseek.sse import extract_content_from_sse, extract_thinking_from_sse

logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================

BASE_URL = "https://chat.deepseek.com"

# DeepSeek 的内部 API 端点（可能变更，如遇 404 请检查）
_SESSION_URL = "/api/v0/chat_session/create"
_COMPLETION_URL = "/api/v0/chat/completion"

_HTTP_TIMEOUT = 300.0  # AI 长回答超时（秒）

# 请求头中使用的 User-Agent
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/148.0.0.0 Safari/537.36"
)

# 新版 API 要求的客户端标识头
_CLIENT_HEADERS = {
    "Origin": BASE_URL,
    "Accept": "*/*",
    "X-App-Version": "2.0.0",
    "X-Client-Bundle-Id": "com.deepseek.chat",
    "X-Client-Locale": "zh_CN",
    "X-Client-Platform": "web",
    "X-Client-Timezone-Offset": "28800",
    "X-Client-Version": "2.0.0",
}


# ============================================================
# 辅助函数
# ============================================================


def _model_type_from_user_model(model: str) -> str:
    """验证并返回 model_type，直接透传。

    Args:
        model: 用户请求中传入的模型 ID

    Returns:
        API 可接受的 model_type 值（default/expert/vision）
    """
    allowed = ("default", "expert", "vision")
    if model not in allowed:
        logger.warning("[模型] 未知模型 '%s'，使用 default", model)
        return "default"
    return model


# ============================================================
# HTTP 客户端
# ============================================================


class _DeepSeekHTTPClient:
    """DeepSeek HTTP 直连客户端。

    负责管理 PoW 浏览器生命周期和 DeepSeek HTTP API 调用。

    生命周期::
        client = _DeepSeekHTTPClient()
        client.start()                    # 读取 token + 启动 PoW 浏览器
        text = client.ask(messages)       # 发送对话
        chunks = client.ask_stream(...)   # 或流式获取
        client.close()                    # 释放资源

    线程安全: 所有公共方法应通过 asyncio.to_thread 调用。
    """

    def __init__(self) -> None:
        self._token: Optional[str] = None
        self._pow_solver: Optional[PowSolver] = None
        self._started = False

    # --------------------------------------------------
    # 生命周期
    # --------------------------------------------------

    def start(self) -> None:
        """初始化客户端。

        步骤:
        1. 从 session 文件读取 token
        2. 启动常驻的 PoW 计算浏览器

        Raises:
            SessionExpiredError: token 不存在或已过期
        """
        self._token = load_token()
        if not self._token:
            raise SessionExpiredError(
                "未检测到登录状态。"
                "请先执行 'uv run python -m gateway.server --mode login'"
            )
        logger.info("[客户端] Token 加载成功 (%d 字符)", len(self._token))

        self._pow_solver = PowSolver()
        self._pow_solver.start()
        self._started = True
        logger.info("[客户端] DeepSeek HTTP 客户端就绪")

    def close(self) -> None:
        """释放所有资源。

        主要包括关闭 PoW 浏览器。
        幂等操作，可多次调用。
        """
        if self._pow_solver:
            self._pow_solver.close()
            self._pow_solver = None
        self._started = False
        logger.info("[客户端] DeepSeek HTTP 客户端已关闭")

    @property
    def is_healthy(self) -> bool:
        """检查客户端是否健康。

        健康条件:
        - 已启动 (start() 已调用)
        - Token 存在
        - PoW 浏览器页面可用

        Returns:
            True 表示可以正常服务
        """
        if not self._started or not self._token:
            return False
        try:
            ok = self._pow_solver is not None and self._pow_solver._page is not None
            return ok
        except Exception:
            return False

    # --------------------------------------------------
    # 核心 API
    # --------------------------------------------------

    def ask(
        self,
        content: str,
        model: str = "default",
        thinking_enabled: bool = True,
        search_enabled: bool = False,
    ) -> str:
        """发送对话消息，返回完整响应文本。

        Args:
            content: 用户输入文本
            model: 模型名（default / expert / vision）
            thinking_enabled: 是否开启深度思考
            search_enabled: 是否开启智能搜索
        """
        if not self._started:
            raise RuntimeError("客户端未启动，请先调用 start()")

        model_type = _model_type_from_user_model(model)
        logger.info(
            "[对话] model=%s think=%s search=%s prompt=%d字符",
            model_type, thinking_enabled, search_enabled, len(content),
        )

        t0 = time.monotonic()
        sid = self._create_session()
        t1 = time.monotonic()
        pow_header = self._pow_solver.solve()
        t2 = time.monotonic()
        text_content, thinking = self._send_completion(
            sid, content, model_type, pow_header,
            thinking_enabled, search_enabled,
        )
        t3 = time.monotonic()

        logger.info(
            "[对话] 完成 (%.1fs | 会话%.1fs PoW%.1fs HTTP%.1fs | %d字符 思考%d字符)",
            t3 - t0, t1 - t0, t2 - t1, t3 - t2, len(text_content), len(thinking),
        )
        return (text_content, thinking)

    def ask_stream(
        self,
        content: str,
        model: str = "default",
        thinking_enabled: bool = True,
        search_enabled: bool = False,
    ) -> list[str]:
        """发送对话消息，返回内容块列表（流式模式用）。

        Args:
            content: 用户输入文本
            model: 模型名
            thinking_enabled: 深度思考
            search_enabled: 智能搜索
        """
        if not self._started:
            raise RuntimeError("客户端未启动，请先调用 start()")

        model_type = _model_type_from_user_model(model)
        logger.info(
            "[流式] model=%s think=%s search=%s prompt=%d字符",
            model_type, thinking_enabled, search_enabled, len(content),
        )

        t0 = time.monotonic()
        sid = self._create_session()
        pow_header = self._pow_solver.solve()
        chunks = self._stream_completion(
            sid, content, model_type, pow_header,
            thinking_enabled, search_enabled,
        )

        logger.info("[流式] 完成 (%.1fs, %d 块)", time.monotonic() - t0, len(chunks))
        return chunks

    # --------------------------------------------------
    # HTTP 请求
    # --------------------------------------------------

    def _create_session(self) -> str:
        """创建新的对话 session。

        POST /api/v0/chat_session/create
        body: {} (空对象)

        Returns:
            session_id (UUID 字符串)

        Raises:
            SessionExpiredError: HTTP 401
            BackendNotAvailableError: 响应格式异常
        """
        url = urljoin(BASE_URL, _SESSION_URL)
        logger.debug("[API] 创建 session ...")

        t0 = time.monotonic()
        resp = httpx.post(
            url,
            json={},
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=_HTTP_TIMEOUT,
        )
        session_elapsed = time.monotonic() - t0

        if resp.status_code == 401:
            logger.warning("[API] 创建 session 返回 401，token 可能已过期 (%.2fs)", session_elapsed)
            raise SessionExpiredError("Token 已过期，请重新登录")

        try:
            resp.raise_for_status()
            data = resp.json()
            sid = data["data"]["biz_data"]["id"]
            logger.debug("[API] session 创建成功: %s (%.2fs)", sid[:8], session_elapsed)
            return sid
        except (KeyError, json.JSONDecodeError) as exc:
            text = resp.text[:500]
            logger.error(
                "[API] session 响应解析失败: %s | 响应: %s (%.2fs)", exc, text, session_elapsed,
            )
            raise BackendNotAvailableError(
                f"创建会话失败: {exc}。响应: {text}"
            ) from exc

    def _send_completion(
        self,
        session_id: str,
        prompt: str,
        model_type: str,
        pow_header: str,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
    ) -> str:
        """发送对话请求并解析 SSE 响应。

        POST /api/v0/chat/completion

        Args:
            session_id: session ID
            prompt: 提示词文本
            model_type: default / expert / vision
            pow_header: x-ds-pow-response 值
            thinking_enabled: 深度思考
            search_enabled: 智能搜索

        Returns:
            响应文本
        """
        url = urljoin(BASE_URL, _COMPLETION_URL)
        body = self._build_completion_body(
            session_id, prompt, model_type, thinking_enabled, search_enabled,
        )
        headers = self._build_completion_headers(
            pow_header, self._token, session_id,
        )

        t0 = time.monotonic()
        logger.debug("[API] 发送对话 (%d 字符) ...", len(prompt))

        resp = httpx.post(
            url, json=body, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        http_elapsed = time.monotonic() - t0

        if resp.status_code == 401:
            logger.warning("[API] 对话请求 401 (%.2fs)", http_elapsed)
            raise SessionExpiredError("Token 已过期，请重新登录")

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("[API] 对话请求失败 HTTP %d (%.2fs)", resp.status_code, http_elapsed)
            raise BackendNotAvailableError(
                f"对话请求失败 (HTTP {resp.status_code}): {resp.text[:300]}"
            ) from exc

        # 解析 SSE，分离 thinking 和 content
        content_parts = []
        thinking_parts = []
        current_type = None
        last_p = ""
        for line in resp.text.split("\n"):
            if not line.startswith("data: "):
                continue
            s = line[6:].strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue
            p = d.get("p") or last_p
            v = d.get("v")
            o = d.get("o")
            if p:
                last_p = p

            # 追踪 fragment 类型
            if p == "response/fragments/-1/content" and isinstance(v, str):
                if o == "APPEND" or not o:
                    if current_type == "THINK":
                        thinking_parts.append(v)
                    else:
                        content_parts.append(v)
            elif p == "response/fragments" and isinstance(v, list) and v:
                last_type = v[-1].get("type") if isinstance(v[-1], dict) else None
                if last_type:
                    current_type = last_type
            elif not p and isinstance(v, dict) and "response" in v:
                resp_obj = v.get("response", {})
                frags = resp_obj.get("fragments", [])
                if frags:
                    current_type = frags[-1].get("type", current_type)

        content = "".join(content_parts).strip()
        thinking = "".join(thinking_parts).strip() if thinking_enabled else ""
        parse_elapsed = time.monotonic() - t0 - http_elapsed
        logger.debug(
            "[API] 对话完成 (%.1fs | HTTP %.1fs 解析 %.2fs | %d字符 思考%d字符)",
            time.monotonic() - t0, http_elapsed, parse_elapsed, len(content), len(thinking),
        )
        return (content, thinking)

    def _stream_completion(
        self,
        session_id: str,
        prompt: str,
        model_type: str,
        pow_header: str,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
    ) -> list[str]:
        """发送对话请求并逐行解析 SSE，返回内容块列表。

        Args:
            session_id: session ID
            prompt: 提示词文本
            model_type: default / expert / vision
            pow_header: x-ds-pow-response 值
            thinking_enabled: 深度思考
            search_enabled: 智能搜索

        Returns:
            内容文本块列表
        """
        url = urljoin(BASE_URL, _COMPLETION_URL)
        body = self._build_completion_body(
            session_id, prompt, model_type, thinking_enabled, search_enabled,
        )
        headers = self._build_completion_headers(
            pow_header, self._token, session_id,
        )

        t0 = time.monotonic()
        logger.debug("[流式API] 发送对话 (%d 字符) ...", len(prompt))

        resp = httpx.post(
            url, json=body, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()

        chunks: list[str] = []
        last_p = ""
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            s = line[6:].strip()
            if not s:
                continue
            try:
                d = json.loads(s)
            except json.JSONDecodeError:
                continue

            p = d.get("p") or last_p
            v = d.get("v")
            o = d.get("o")
            if p:
                last_p = p

            if isinstance(v, str) and p == "response/fragments/-1/content":
                if o == "APPEND" or not o:
                    chunks.append(v)

        logger.debug(
            "[流式API] 完成 (%.1fs, %d 块)", time.monotonic() - t0, len(chunks),
        )
        return chunks

    def _stream_to_queue(
        self,
        session_id: str,
        prompt: str,
        model_type: str,
        pow_header: str,
        output_queue: queue.Queue,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
    ) -> None:
        """在线程中完成完整流式请求：创建 session → PoW → HTTP 流式读取。

        每得到一个 chunk 就放入 output_queue。
        结束后放入 None 作为结束标记。
        """
        try:
            # 创建 session（如果未传入）
            if not session_id:
                sid_resp = httpx.post(
                    urljoin(BASE_URL, _SESSION_URL), json={},
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=_HTTP_TIMEOUT,
                )
                if sid_resp.status_code == 401:
                    raise SessionExpiredError("Token 已过期，请重新登录")
                sid_resp.raise_for_status()
                session_id = sid_resp.json()["data"]["biz_data"]["id"]

            # 获取 PoW（如果未传入）
            if not pow_header:
                pow_header = self._pow_solver.solve()

            # 发起流式 HTTP 请求
            url = urljoin(BASE_URL, _COMPLETION_URL)
            body = self._build_completion_body(
                session_id, prompt, model_type, thinking_enabled, search_enabled,
            )
            headers = self._build_completion_headers(pow_header, self._token, session_id)

            with httpx.Client(timeout=_HTTP_TIMEOUT) as client:
                with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code == 401:
                        raise SessionExpiredError("Token 已过期，请重新登录")
                    resp.raise_for_status()

                    last_p = ""
                    current_type = None
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        s = line[6:].strip()
                        if not s:
                            continue
                        try:
                            d = json.loads(s)
                        except json.JSONDecodeError:
                            continue

                        p = d.get("p") or last_p
                        v = d.get("v")
                        o = d.get("o")
                        if p:
                            last_p = p

                        # 追踪当前 fragment 类型（THINK / RESPONSE）
                        if isinstance(v, str) and p == "response/fragments/-1/content":
                            if o == "APPEND" or not o:
                                is_think = current_type == "THINK"
                                output_queue.put((v, is_think))
                        elif p == "response/fragments" and isinstance(v, list) and v:
                            last_type = v[-1].get("type") if isinstance(v[-1], dict) else None
                            if last_type:
                                current_type = last_type
                        elif not p and isinstance(v, dict) and "response" in v:
                            resp_obj = v.get("response", {})
                            frags = resp_obj.get("fragments", [])
                            if frags:
                                current_type = frags[-1].get("type", current_type)
        except Exception as exc:
            logger.error("[流式] 线程错误: %s", exc)
            output_queue.put(("[error]", False))
        finally:
            output_queue.put((None, False))

    # --------------------------------------------------
    # 请求体/头构建（ask 和 ask_stream 复用）
    # --------------------------------------------------

    @staticmethod
    def _build_completion_body(
        session_id: str, prompt: str, model_type: str,
        thinking_enabled: bool = True, search_enabled: bool = False,
    ) -> dict:
        """构建 /api/v0/chat/completion 的请求体。"""
        return {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }

    @staticmethod
    def _build_completion_headers(pow_header: str, token: str, session_id: str) -> dict:
        """构建 /api/v0/chat/completion 的请求头。"""
        return {
            "Authorization": f"Bearer {token}",
            "X-Ds-Pow-Response": pow_header,
            "Referer": f"{BASE_URL}/a/chat/s/{session_id}",
            "User-Agent": _USER_AGENT,
            **_CLIENT_HEADERS,
        }
