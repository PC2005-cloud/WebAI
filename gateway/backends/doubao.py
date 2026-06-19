"""豆包后端适配器（待实现）。

TODO: Phase 3 实现内容
-----------------------
本适配器计划实现豆包网页版的程序化调用。

调研方向:
1. 打开豆包网页版 → F12 → Network → 观察请求
2. 寻找 chat/completion、conversation 等关键词
3. 分析请求头、响应格式（SSE? JSON?）
4. 如果有 PoW 类验证，参考 deepseek 的 PowSolver 实现

备选方案:
    A. HTTP 直连（如果有类似 DeepSeek 的内部 API）— 首选
    B. Playwright 全浏览器自动化 — 慢但可靠

实现参考: gateway/deepseek/ 下的模块结构和 gateway/backends/deepseek.py
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel, Field

from gateway.backends import register
from gateway.backends.base import BaseBackend
from gateway.schemas import Message, MessageList


class ChatRequest(BaseModel):
    """对话请求体。"""

    messages: list[Message] = Field(
        description="OpenAI 格式的消息列表，如 [{\"role\": \"user\", \"content\": \"你好\"}]"
    )
    model: str = Field("default", description="模型 ID")
    stream: bool = Field(False, description="是否使用 SSE 流式响应")


@register("doubao")
class DoubaoBackend(BaseBackend):
    """豆包后端适配器（待实现）。

    当前返回 NotImplementedError。
    实现后请更新 gateway/backends/__init__.py 中的后端列表。
    """

    @property
    def name(self) -> str:
        return "doubao"

    @property
    def display_name(self) -> str:
        return "豆包"

    @property
    def models(self) -> list[str]:
        return ["default"]

    def register_routes(self, router: APIRouter) -> None:
        prefix = f"/v1/{self.name}"
        router.add_api_route(
            f"{prefix}/chat",
            self.chat_endpoint,
            methods=["POST"],
            summary="发送对话消息（TODO）",
            description="**豆包后端尚未实现**，此端点当前返回 501 Not Implemented。",
            tags=["豆包"],
        )
        router.add_api_route(
            f"{prefix}/models",
            self.models_endpoint,
            methods=["GET"],
            summary="列出可用模型",
            description="返回当前豆包后端支持的模型 ID 列表（当前仅返回占位值）。",
            tags=["豆包"],
        )

    async def chat_endpoint(self, req: ChatRequest):
        """POST /v1/doubao/chat（待实现）"""
        raise NotImplementedError("Phase 3 实现")

    async def models_endpoint(self):
        """GET /v1/doubao/models"""
        return {"models": self.models}

    async def chat(
        self,
        messages: MessageList,
        model: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        raise NotImplementedError("Phase 3 实现")

    async def check_health(self) -> bool:
        raise NotImplementedError("Phase 3 实现")
