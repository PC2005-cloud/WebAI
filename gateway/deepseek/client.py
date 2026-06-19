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
import time
from typing import Optional
from urllib.parse import urljoin

import httpx

from core.exceptions import (
    BackendNotAvailableError,
    PowTimeoutError,
    SessionExpiredError,
)
from gateway.deepseek.pow import PowSolver
from gateway.deepseek.session import load_token
from gateway.deepseek.sse import extract_content_from_sse

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
    "Chrome/148.0.0.0 Safari/537.36"
)


# ============================================================
# 辅助函数
# ============================================================


def _build_prompt(messages: list[dict]) -> str:
    """将 OpenAI 格式的 messages 列表拼接为纯文本提示词。

    每个消息按角色添加标签，全部拼接后发给 DeepSeek 网页版 API。
    标签名不重要（AI 网页版没有指令微调来识别特定标签），
    重要的是内容和顺序的连贯性。

    Args:
        messages: OpenAI 格式的消息列表

    Returns:
        拼接后的纯文本
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[系统指令]\n{content}")
        elif role == "user":
            parts.append(f"[用户]\n{content}")
        elif role == "assistant":
            parts.append(f"[助手]\n{content}")
        elif role == "tool":
            parts.append(f"[工具执行结果]\n{content}")
        else:
            parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def _model_type_from_user_model(model: str) -> str:
    """将用户可见的模型名映射为 API 的 model_type 参数。

    映射关系:
        "default"  → "default"  （快速模式）
        "thinking" → "expert"   （专家模式，旧名 R1/思考）
        其他       → "default"  （默认）

    Args:
        model: 用户请求中传入的模型 ID

    Returns:
        API 可接受的 model_type 值
    """
    return "expert" if model == "thinking" else "default"


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

    def ask(self, messages: list[dict], model: str = "default") -> str:
        """发送对话消息，返回完整响应文本。

        这是非流式接口——等待完整响应后一次性返回。

        Args:
            messages: OpenAI 格式的消息列表
            model: 模型名（"default"/"thinking"）

        Returns:
            响应文本（已去除首尾空白）

        Raises:
            RuntimeError: 客户端未调用 start()
            SessionExpiredError: token 过期或无效
            PowTimeoutError: PoW 计算超时
            BackendNotAvailableError: DeepSeek 后端返回错误
        """
        if not self._started:
            raise RuntimeError("客户端未启动，请先调用 start()")

        prompt = _build_prompt(messages)
        model_type = _model_type_from_user_model(model)
        logger.info(
            "[对话] 开始请求 model=%s messages=%d条 prompt=%d字符",
            model_type, len(messages), len(prompt),
        )

        t_start = time.monotonic()

        # 1. 创建会话
        sid = self._create_session()
        t_session = time.monotonic()

        # 2. 获取 PoW
        pow_header = self._pow_solver.solve()
        t_pow = time.monotonic()

        # 3. 发送对话
        content = self._send_completion(sid, prompt, model_type, pow_header)
        t_done = time.monotonic()

        logger.info(
            "[对话] 完成 | 会话=%.1fs PoW=%.1fs HTTP=%.1fs 总计=%.1fs | 响应=%d字符",
            t_session - t_start,
            t_pow - t_session,
            t_done - t_pow,
            t_done - t_start,
            len(content),
        )
        return content

    def ask_stream(
        self, messages: list[dict], model: str = "default"
    ) -> list[str]:
        """发送对话消息，返回内容块列表（流式模式用）。

        返回预收集的 list[str] 而非实时生成器，
        因为 httpx 同步请求必须先等完整响应才能解析。

        FastAPI 端拿到列表后再用 SSE EventSourceResponse 逐块发送。

        Args:
            messages: OpenAI 格式的消息列表
            model: 模型名

        Returns:
            内容文本块列表
        """
        if not self._started:
            raise RuntimeError("客户端未启动，请先调用 start()")

        prompt = _build_prompt(messages)
        model_type = _model_type_from_user_model(model)
        logger.info(
            "[流式] 开始请求 model=%s messages=%d条 prompt=%d字符",
            model_type, len(messages), len(prompt),
        )

        t_start = time.monotonic()

        # 1-2. 创建会话 + PoW
        sid = self._create_session()
        pow_header = self._pow_solver.solve()
        t_ready = time.monotonic()

        # 3. 发送对话并逐行流式解析
        chunks = self._stream_completion(sid, prompt, model_type, pow_header)
        t_done = time.monotonic()

        logger.info(
            "[流式] 完成 | 准备=%.1fs 传输=%.1fs 总计=%.1fs | %d块 %d字符",
            t_ready - t_start,
            t_done - t_ready,
            t_done - t_start,
            len(chunks),
            sum(len(c) for c in chunks),
        )
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

        resp = httpx.post(
            url,
            json={},
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=_HTTP_TIMEOUT,
        )

        if resp.status_code == 401:
            logger.warning("[API] 创建 session 返回 401，token 可能已过期")
            raise SessionExpiredError("Token 已过期，请重新登录")

        try:
            resp.raise_for_status()
            data = resp.json()
            sid = data["data"]["biz_data"]["id"]
            logger.debug("[API] session 创建成功: %s", sid[:8])
            return sid
        except (KeyError, json.JSONDecodeError) as exc:
            text = resp.text[:500]
            logger.error(
                "[API] session 响应解析失败: %s | 响应: %s", exc, text
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
    ) -> str:
        """发送对话请求并解析 SSE 响应，返回完整文本。

        POST /api/v0/chat/completion

        Args:
            session_id: 由上一步 _create_session 获得
            prompt: 拼接好的提示词文本
            model_type: "default" / "expert" / "vision"
            pow_header: x-ds-pow-response 值

        Returns:
            响应文本
        """
        url = urljoin(BASE_URL, _COMPLETION_URL)
        body = self._build_completion_body(session_id, prompt, model_type)
        headers = self._build_completion_headers(pow_header, self._token)

        t0 = time.monotonic()
        logger.debug("[API] 发送对话 (%d 字符) ...", len(prompt))

        resp = httpx.post(
            url, json=body, headers=headers, timeout=_HTTP_TIMEOUT,
        )

        if resp.status_code == 401:
            raise SessionExpiredError("Token 已过期，请重新登录")

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise BackendNotAvailableError(
                f"对话请求失败 (HTTP {resp.status_code}): {resp.text[:300]}"
            ) from exc

        # 解析 SSE 响应
        content = extract_content_from_sse(resp.text)
        elapsed = time.monotonic() - t0
        logger.debug(
            "[API] 对话完成 (%.1fs, 响应 %d 字节, 提取 %d 字符)",
            elapsed, len(resp.text), len(content),
        )
        return content

    def _stream_completion(
        self,
        session_id: str,
        prompt: str,
        model_type: str,
        pow_header: str,
    ) -> list[str]:
        """发送对话请求并逐行解析 SSE，返回内容块列表。

        与 _send_completion 的区别：
        - 使用 resp.iter_lines() 逐行读取
        - 收集每块内容后返回列表（供 EventSourceResponse 使用）

        Args:
            同上 _send_completion

        Returns:
            内容文本块列表
        """
        url = urljoin(BASE_URL, _COMPLETION_URL)
        body = self._build_completion_body(session_id, prompt, model_type)
        headers = self._build_completion_headers(pow_header, self._token)

        t0 = time.monotonic()
        logger.debug("[流式API] 发送对话 (%d 字符) ...", len(prompt))

        resp = httpx.post(
            url, json=body, headers=headers, timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()

        # 逐行解析 SSE
        chunks: list[str] = []
        last_p = ""

        for line in resp.iter_lines(decode_unicode=True):
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

            if isinstance(v, str) and p == "response/content":
                if o == "APPEND" or not o:
                    chunks.append(v)

        elapsed = time.monotonic() - t0
        logger.debug(
            "[流式API] 完成 (%.1fs, %d 块, %d 字符)",
            elapsed,
            len(chunks),
            sum(len(c) for c in chunks),
        )
        return chunks

    # --------------------------------------------------
    # 请求体/头构建（ask 和 ask_stream 复用）
    # --------------------------------------------------

    @staticmethod
    def _build_completion_body(
        session_id: str, prompt: str, model_type: str
    ) -> dict:
        """构建 /api/v0/chat/completion 的请求体。

        Args:
            session_id: 对话 session ID
            prompt: 提示词文本
            model_type: 模型类型

        Returns:
            请求体 dict
        """
        return {
            "chat_session_id": session_id,
            "parent_message_id": None,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": [],
            "thinking_enabled": True,
            "search_enabled": True,
            "action": None,
            "preempt": False,
        }

    @staticmethod
    def _build_completion_headers(pow_header: str, token: str) -> dict:
        """构建 /api/v0/chat/completion 的请求头。

        Args:
            pow_header: x-ds-pow-response 值
            token: Bearer token

        Returns:
            请求头 dict
        """
        return {
            "Authorization": f"Bearer {token}",
            "X-Ds-Pow-Response": pow_header,
            "User-Agent": _USER_AGENT,
        }
