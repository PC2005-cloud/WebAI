"""DeepSeek 后端适配器。

实现方案（HTTP 直连）
--------------------
本适配器不走浏览器界面，而是直接调 DeepSeek 内部 HTTP API。
PoW 计算仍需要隐藏浏览器，但每次只需 ∼10s 即可复用。

调用链路:
    POST /v1/deepseek/chat
      → DeepSeekBackend.chat_endpoint()
        → _ensure_client()        # 首次调用时在线程中初始化
          → _DeepSeekHTTPClient.start()
            → load_token()
            → PowSolver.start()    # 启动常驻 headless 浏览器
        → _handle_normal / _handle_stream
          → asyncio.to_thread(client.ask/ask_stream)
            → _create_session()    # POST /api/v0/chat_session/create
            → PowSolver.solve()    # 触发 PoW 计算
            → _send_completion()   # POST /api/v0/chat/completion
            → extract_content_from_sse()  # 解析 SSE 响应

线程安全
--------
Playwright 和 httpx 都是同步库，不能在 FastAPI async 事件循环中直接调用。
所有同步操作都通过 asyncio.to_thread 委托到线程池执行。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.exceptions import (
    BackendNotAvailableError,
    PowTimeoutError,
    SessionExpiredError,
)
from gateway.backends import register
from gateway.backends.base import BaseBackend
from gateway.deepseek.client import _DeepSeekHTTPClient
from gateway.schemas import Message, MessageList

logger = logging.getLogger(__name__)


# ============================================================
# 请求/响应模型
# ============================================================


class ChatRequest(BaseModel):
    """对话请求体。"""

    messages: list[Message] = Field(
        description="""OpenAI 格式的消息列表。

每条消息包含 role 和 content 字段：
- **role**: "system" | "user" | "assistant" | "tool"
- **content**: 消息文本

示例：
```json
[
  {"role": "system", "content": "你是一个数学助手"},
  {"role": "user",   "content": "1+1等于几？"}
]
```"""
    )
    model: str = Field(
        "default",
        description="模型 ID，当前仅支持 `default`",
    )
    stream: bool = Field(
        False,
        description="是否使用 SSE 流式响应。\n- `false`：一次性返回完整文本\n- `true`：逐块推送内容",
    )


class ChatResponse(BaseModel):
    """非流式对话响应体。"""

    id: str = Field(description="对话标识，固定为 `chat-deepseek`")
    content: str = Field(description="AI 回复文本")
    finish_reason: str = Field(
        "stop",
        description="结束原因，`stop` 表示正常完成",
    )


# ============================================================
# 后端适配器
# ============================================================


@register("deepseek")
class DeepSeekBackend(BaseBackend):
    """DeepSeek 后端适配器。

    首次调用 chat 时在线程池中懒初始化 _DeepSeekHTTPClient
    （含 PoW 浏览器启动，耗时 ∼15s）。
    后续调用复用已初始化的客户端。
    """

    def __init__(self) -> None:
        self._client: _DeepSeekHTTPClient | None = None
        self._init_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "deepseek"

    @property
    def display_name(self) -> str:
        return "DeepSeek"

    @property
    def models(self) -> list[str]:
        return ["default"]

    def register_routes(self, router: APIRouter) -> None:
        prefix = f"/v1/{self.name}"

        router.add_api_route(
            f"{prefix}/chat",
            self.chat_endpoint,
            methods=["POST"],
            summary="发送对话消息",
            description="""
    向 DeepSeek 发送消息并获取 AI 回复。

    ## 非流式（stream=false 默认）
    发送完整消息后等待 AI 回复完成，一次性返回结果。

    请求体示例：
    ```json
    {
      "messages": [
        {"role": "system", "content": "你是一个数学助手"},
        {"role": "user", "content": "1+1等于几？"}
      ],
      "model": "default",
      "stream": false
    }
    ```

    响应示例：
    ```json
    {
      "code": 1,
      "message": "success",
      "data": {
        "id": "chat-deepseek",
        "content": "1+1等于2。",
        "finish_reason": "stop"
      }
    }
    ```

    ## 流式（stream=true）
    使用 SSE (Server-Sent Events) 逐块推送内容。

    请求体：
    ```json
    {
      "messages": [{"role": "user", "content": "写一首诗"}],
      "model": "default",
      "stream": true
    }
    ```

    SSE 推送格式（非标准 OpenAPI SSE）：
    ```
    data: {"type": "content", "content": "床前"}
    data: {"type": "content", "content": "明月"}
    data: {"type": "done", "finish_reason": "stop"}
    ```

    ## 错误码
    | 状态码 | 含义 | 处理方式 |
    |--------|------|---------|
    | 200 | 成功 | 正常解析 data 字段 |
    | 402 | 未登录或 token 过期 | 执行 --mode login 重新登录 |
    | 502 | DeepSeek 后端返回错误 | 稍后重试 |
    | 503 | PoW 验证超时 | 稍后重试 |
    """,
            tags=["DeepSeek"],
            response_model=ChatResponse,
            responses={
                200: {"description": "对话成功，返回 AI 回复内容"},
                402: {
                    "description": "未登录或登录态已过期",
                    "content": {
                        "application/json": {
                            "example": {
                                "error": "session_expired",
                                "message": "未检测到登录状态。请先执行 --mode login",
                                "hint": "请先执行 'uv run python -m gateway.server --mode login'",
                            }
                        }
                    },
                },
                502: {
                    "description": "DeepSeek 后端服务异常",
                    "content": {
                        "application/json": {
                            "example": {
                                "error": "backend_unavailable",
                                "message": "对话请求失败 (HTTP 500): ...",
                            }
                        }
                    },
                },
                503: {
                    "description": "PoW 验证服务暂时不可用",
                    "content": {
                        "application/json": {
                            "example": {
                                "error": "pow_timeout",
                                "message": "PoW 计算超时（30s）",
                                "hint": "请稍后重试",
                            }
                        }
                    },
                },
            },
        )

        router.add_api_route(
            f"{prefix}/models",
            self.models_endpoint,
            methods=["GET"],
            summary="列出可用模型",
            description="返回当前 DeepSeek 后端支持的模型 ID 列表。",
            tags=["DeepSeek"],
        )

    # --------------------------------------------------
    # 端点实现
    # --------------------------------------------------

    async def chat_endpoint(self, req: ChatRequest):
        """POST /v1/deepseek/chat — 发送对话。"""
        # 1. 确保客户端已初始化（首次需启动 PoW 浏览器）
        try:
            await self._ensure_client()
        except SessionExpiredError as exc:
            logger.warning("对话请求被拒: 未登录 (%s)", exc)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=402,
                content={
                    "error": "session_expired",
                    "message": str(exc),
                    "hint": "请先执行 'uv run python -m gateway.server --mode login'",
                },
            )

        # 2. 处理对话
        logger.info(
            "[请求] model=%s stream=%s messages=%d条",
            req.model, req.stream, len(req.messages),
        )
        try:
            if req.stream:
                return await self._handle_stream(req)
            return await self._handle_normal(req)
        except SessionExpiredError as exc:
            logger.warning("Token 过期: %s", exc)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=402,
                content={"error": "session_expired", "message": str(exc)},
            )
        except PowTimeoutError as exc:
            logger.error("PoW 超时: %s", exc)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=503,
                content={
                    "error": "pow_timeout",
                    "message": str(exc),
                    "hint": "DeepSeek 验证服务暂时不可用，请稍后重试",
                },
            )
        except BackendNotAvailableError as exc:
            logger.error("后端返回错误: %s", exc)
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=502,
                content={
                    "error": "backend_unavailable",
                    "message": str(exc),
                },
            )

    async def models_endpoint(self):
        """GET /v1/deepseek/models — 返回可用模型列表。"""
        return {"models": self.models}

    # --------------------------------------------------
    # BaseBackend 接口实现
    # --------------------------------------------------

    async def chat(
        self,
        messages: MessageList,
        model: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """实现 BaseBackend 接口，供内部或其他模块调用。"""
        await self._ensure_client()
        if stream:
            return self._stream_response(messages, model)
        return await asyncio.to_thread(self._client.ask, messages, model)

    async def check_health(self) -> bool:
        """检查 DeepSeek 后端是否可用。

        检查项:
        - Token 是否存在且有效
        - PoW 浏览器是否正常运行
        """
        try:
            if self._client is None:
                logger.debug("[健康检查] 客户端未初始化")
                return False
            healthy = self._client.is_healthy
            logger.debug("[健康检查] DeepSeek 状态: %s", "正常" if healthy else "异常")
            return healthy
        except Exception as exc:
            logger.warning("[健康检查] 检查失败: %s", exc)
            return False

    # --------------------------------------------------
    # 内部方法
    # --------------------------------------------------

    async def _ensure_client(self) -> None:
        """确保 DeepSeek HTTP 客户端已初始化（线程安全，仅首次执行）。

        使用双重检查锁定（Double-Checked Locking）模式:
        1. 先判断 self._client 是否 None（无锁）
        2. 拿到 async Lock 后再次判断（避免竞态）
        """
        if self._client is not None:
            return

        async with self._init_lock:
            if self._client is not None:
                return

            logger.info("[初始化] 首次请求，启动 DeepSeek 客户端 ...")
            client = _DeepSeekHTTPClient()
            await asyncio.to_thread(client.start)
            self._client = client
            logger.info("[初始化] DeepSeek 客户端就绪（PoW 浏览器已常驻）")

    async def _handle_normal(self, req: ChatRequest) -> dict:
        """处理非流式请求。

        在线程池中运行同步的 client.ask()，
        返回统一格式的响应 JSON。
        """
        content = await asyncio.to_thread(
            self._client.ask, req.messages, req.model
        )
        logger.info("[响应] 非流式完成, 长度=%d字符", len(content))
        return {
            "id": "chat-deepseek",
            "content": content,
            "finish_reason": "stop",
        }

    async def _handle_stream(self, req: ChatRequest) -> EventSourceResponse:
        """处理流式请求。

        先在线程池中收集所有内容块（同步 httpx 只能这样），
        再用 SSE EventSourceResponse 逐块发送给客户端。
        """
        chunks = await asyncio.to_thread(
            self._client.ask_stream, req.messages, req.model
        )
        logger.info("[响应] 流式完成, %d个块", len(chunks))

        async def event_generator():
            for chunk in chunks:
                yield {"type": "content", "content": chunk}
            yield {"type": "done", "finish_reason": "stop"}

        return EventSourceResponse(event_generator())

    async def _stream_response(
        self,
        messages: MessageList,
        model: str,
    ) -> AsyncGenerator[str, None]:
        """异步生成器接口——逐块 yield 内容给上游调用者。"""
        chunks = await asyncio.to_thread(
            self._client.ask_stream, messages, model
        )
        for chunk in chunks:
            yield chunk
