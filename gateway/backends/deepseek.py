"""DeepSeek 后端适配器。

实现方案（HTTP 直连）
--------------------
本适配器不走浏览器界面，而是直接调 DeepSeek 内部 HTTP API。
PoW 计算仍需要隐藏浏览器，但每次只需 ∼10s 即可复用。

调用链路:
    POST /v1/deepseek/chat
      → DeepSeekBackend.chat_endpoint()
        → _ensure_client()
          → _DeepSeekHTTPClient.start()
            → load_token()
            → PowSolver.start()
        → _handle_normal / _handle_stream
          → asyncio.to_thread(client.ask/ask_stream)
            → _create_session()
            → PowSolver.solve()
            → _send_completion()
            → extract_content_from_sse()

线程安全
--------
Playwright 和 httpx 都是同步库，不能在 FastAPI async 事件循环中直接调用。
所有同步操作都通过 asyncio.to_thread 委托到线程池执行。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter
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

logger = logging.getLogger(__name__)


# ============================================================
# 请求/响应模型
# ============================================================


class ChatRequest(BaseModel):
    """对话请求体。"""

    content: str = Field(
        description="用户输入文本",
    )
    model: str = Field(
        "default",
        description="模型: `default`（快速）/ `expert`（专家）/ `vision`（识图）",
    )
    thinking_enabled: bool = Field(
        True,
        description="是否开启深度思考",
    )
    search_enabled: bool = Field(
        False,
        description="是否开启智能搜索（仅快速模式有效）",
    )
    stream: bool = Field(
        False,
        description="是否使用 SSE 流式响应",
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

    首次调用 chat 时懒初始化 _DeepSeekHTTPClient（含 PoW 浏览器）。
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
        return ["default", "expert", "vision"]

    def register_routes(self, router: APIRouter) -> None:
        prefix = f"/v1/{self.name}"

        router.add_api_route(
            f"{prefix}/chat",
            self.chat_endpoint,
            methods=["POST"],
            summary="发送对话消息",
            description="""
    向 DeepSeek 发送消息并获取 AI 回复。

    请求体示例：
    ```json
    {
      "content": "你好",
      "model": "default",
      "thinking_enabled": true,
      "search_enabled": false,
      "stream": false
    }
    ```

    响应示例：
    ```json
    {
      "id": "chat-deepseek",
      "content": "你好！有什么可以帮你的？",
      "finish_reason": "stop"
    }
    ```

    ## 模型说明
    | model | 说明 | 深度思考 | 智能搜索 |
    |-------|------|---------|---------|
    | default | 快速模式 | ✅ | ✅ |
    | expert | 专家模式 | ✅ | ❌ |
    | vision | 识图模式 | ✅ | ❌ |
    """,
            tags=["DeepSeek"],
            response_model=ChatResponse,
            responses={
                200: {"description": "对话成功"},
                402: {"description": "未登录或 token 过期"},
                502: {"description": "DeepSeek 后端服务异常"},
                503: {"description": "PoW 验证超时"},
            },
        )

        router.add_api_route(
            f"{prefix}/models",
            self.models_endpoint,
            methods=["GET"],
            summary="列出可用模型",
            tags=["DeepSeek"],
        )

    # --------------------------------------------------
    # 端点实现
    # --------------------------------------------------

    async def chat_endpoint(self, req: ChatRequest):
        """POST /v1/deepseek/chat"""
        _t0 = time.monotonic()
        logger.info(
            "[%s] model=%s content=%s字节 think=%s search=%s stream=%s",
            "chat", req.model, len(req.content),
            req.thinking_enabled, req.search_enabled, req.stream,
        )

        try:
            await self._ensure_client()
        except SessionExpiredError as exc:
            logger.warning("未登录 (%s)", exc)
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=402,
                content={"error": "session_expired", "message": str(exc)},
            )

        try:
            result = await (self._handle_stream(req) if req.stream else self._handle_normal(req))
            logger.info("[%s] 完成 %.1fs", "stream" if req.stream else "chat", time.monotonic() - _t0)
            return result
        except SessionExpiredError as exc:
            from fastapi.responses import JSONResponse
            logger.warning("token过期 %.1fs", time.monotonic() - _t0)
            return JSONResponse(status_code=402, content={"error": "session_expired", "message": str(exc)})
        except PowTimeoutError as exc:
            from fastapi.responses import JSONResponse
            logger.error("PoW超时 %.1fs", time.monotonic() - _t0)
            return JSONResponse(status_code=503, content={"error": "pow_timeout", "message": str(exc)})
        except BackendNotAvailableError as exc:
            from fastapi.responses import JSONResponse
            logger.error("后端错误 %.1fs", time.monotonic() - _t0)
            return JSONResponse(status_code=502, content={"error": "backend_unavailable", "message": str(exc)})

    async def models_endpoint(self):
        """GET /v1/deepseek/models"""
        return {"models": self.models}

    # --------------------------------------------------
    # BaseBackend 接口实现
    # --------------------------------------------------

    async def chat(
        self,
        content: str,
        model: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """实现 BaseBackend 接口。"""
        await self._ensure_client()
        if stream:
            return self._stream_response(content, model)
        return await asyncio.to_thread(self._client.ask, content, model)

    async def check_health(self) -> bool:
        try:
            if self._client is None:
                return False
            return self._client.is_healthy
        except Exception:
            return False

    # --------------------------------------------------
    # 内部方法
    # --------------------------------------------------

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return
        async with self._init_lock:
            if self._client is not None:
                return
            logger.info("[初始化] 启动 DeepSeek 客户端 ...")
            client = _DeepSeekHTTPClient()
            await asyncio.to_thread(client.start)
            self._client = client
            logger.info("[初始化] DeepSeek 客户端就绪")

    async def _handle_normal(self, req: ChatRequest) -> dict:
        content = await asyncio.to_thread(
            self._client.ask, req.content, req.model,
            req.thinking_enabled, req.search_enabled,
        )
        logger.info("[响应] 非流式完成, %d字符", len(content))
        return {
            "id": "chat-deepseek",
            "content": content,
            "finish_reason": "stop",
        }

    async def _handle_stream(self, req: ChatRequest) -> EventSourceResponse:
        chunks = await asyncio.to_thread(
            self._client.ask_stream, req.content, req.model,
            req.thinking_enabled, req.search_enabled,
        )
        logger.info("[响应] 流式完成, %d块", len(chunks))

        async def event_generator():
            for chunk in chunks:
                yield {"type": "content", "content": chunk}
            yield {"type": "done", "finish_reason": "stop"}

        return EventSourceResponse(event_generator())

    async def _stream_response(
        self, content: str, model: str,
    ) -> AsyncGenerator[str, None]:
        chunks = await asyncio.to_thread(self._client.ask_stream, content, model)
        for chunk in chunks:
            yield chunk
