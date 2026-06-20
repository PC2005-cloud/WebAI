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
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

# Playwright 不是线程安全的，使用专用单线程执行所有 PoW 操作
_pow_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pow")

from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse

from core.response import Result
from gateway.backends import register
from gateway.backends.base import BaseBackend
from gateway.deepseek.client import _DeepSeekHTTPClient, _model_type_from_user_model

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
    auto_clean: bool = Field(
        True,
        description="对话完成后是否自动删除该会话（不清除聊天记录）",
    )
    stream: bool = Field(
        False,
        description="是否使用 SSE 流式响应",
    )


class DeleteRequest(BaseModel):
    """删除对话请求体。"""

    session_id: str = Field(description="要删除的对话 session ID")


class ChatResponse(BaseModel):
    """对话响应体。"""

    id: str = Field(description="对话 session ID，可用于删除接口")
    content: str = Field(description="AI 回复文本")
    thinking: str | None = Field(
        None,
        description="深度思考的思维链内容，未开启思考时为 null",
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
            response_model=Result[ChatResponse],
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
            response_model=Result,
        )

        router.add_api_route(
            f"{prefix}/session/delete",
            self.delete_session_endpoint,
            methods=["POST"],
            summary="删除对话记录",
            description="通过 session ID 删除对话记录，清理聊天列表。",
            tags=["DeepSeek"],
            response_model=Result,
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

        # 校验模型名
        allowed_models = ("default", "expert", "vision")
        if req.model not in allowed_models:
            raise ValueError(f"不支持的模型 '{req.model}'，可选: {', '.join(allowed_models)}")

        # 校验参数组合
        if req.model != "default" and req.search_enabled:
            raise ValueError(f"'{req.model}' 模式不支持智能搜索")

        await self._ensure_client()
        result = await (self._handle_stream(req) if req.stream else self._handle_normal(req))
        logger.info("[%s] 完成 %.1fs", "stream" if req.stream else "chat", time.monotonic() - _t0)
        return result

    async def models_endpoint(self):
        """GET /v1/deepseek/models"""
        from core.response import Result
        return Result.success(self.models)

    async def delete_session_endpoint(self, req: DeleteRequest):
        """POST /v1/deepseek/session/delete"""
        from core.response import Result
        try:
            if self._client is not None:
                self._client.delete_session(req.session_id)
            else:
                # 直接调用 API 删除
                import httpx
                from gateway.deepseek.session import load_token
                token = load_token()
                if token:
                    httpx.post(
                        "https://chat.deepseek.com/api/v0/chat_session/delete",
                        json={"chat_session_id": req.session_id},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=10,
                    )
            return Result.success(None)
        except Exception as exc:
            return Result.error(str(exc))

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
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_pow_executor, self._client.ask, content, model)

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
            client = _DeepSeekHTTPClient(pow_executor=_pow_executor)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(_pow_executor, client.start)
            self._client = client
            logger.info("[初始化] DeepSeek 客户端就绪")

    async def _handle_normal(self, req: ChatRequest) -> dict:
        from core.response import Result

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _pow_executor, self._client.ask, req.content, req.model,
            req.thinking_enabled, req.search_enabled, req.auto_clean,
        )
        content, thinking, session_id = result if isinstance(result, tuple) and len(result) == 3 else (result, "", "")
        logger.info("[响应] 非流式完成, %d字符 思考%d字符", len(content), len(thinking))

        return Result.success(
            ChatResponse(id=session_id, content=content, thinking=thinking or None)
        )

    async def _handle_stream(self, req: ChatRequest) -> StreamingResponse:
        import queue as q_mod
        loop = asyncio.get_running_loop()

        _t0 = time.monotonic()
        session_id = await asyncio.to_thread(self._client._create_session)
        _t1 = time.monotonic()
        pow_header = self._client._get_pow()
        _t2 = time.monotonic()
        model_type = _model_type_from_user_model(req.model)
        logger.info("[流式] 会话%.1fs PoW%.1fs 总计%.1fs",
                     _t1 - _t0, _t2 - _t1, _t2 - _t0)

        sync_queue = q_mod.Queue()
        thread = threading.Thread(
            target=self._client._stream_to_queue,
            args=(
                session_id, req.content, model_type, pow_header,
                sync_queue, req.thinking_enabled, req.search_enabled,
                req.auto_clean,
            ),
            daemon=True,
        )
        thread.start()

        async def event_generator():
            chunk_count = 0
            first_t = None
            while True:
                item = await asyncio.to_thread(sync_queue.get)
                chunk, is_think = item if isinstance(item, tuple) else (item, False)
                if chunk is None:
                    break
                if first_t is None:
                    first_t = time.monotonic()
                chunk_count += 1
                data = {
                    "id": "chatcmpl-deepseek",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": chunk, "thinking": is_think}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

            stream_t = time.monotonic() - (first_t or _t0)
            logger.info("[流式] 完成: %d块 流式%.1fs", chunk_count, stream_t)
            data = {
                "id": "chatcmpl-deepseek",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"x-session-id": session_id},
        )

    async def _stream_response(
        self, content: str, model: str,
    ) -> AsyncGenerator[str, None]:
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(_pow_executor, self._client.ask_stream, content, model)
        for chunk in chunks:
            yield chunk
