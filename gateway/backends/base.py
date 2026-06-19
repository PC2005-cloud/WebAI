"""BaseBackend — 所有后端适配器的抽象接口。

新后端必须继承 BaseBackend 并实现所有 @abstractmethod 方法。
注册方式见 backends/__init__.py 的 @register 装饰器。

接口概览:
    name          — 后端唯一标识，也是路由前缀
    display_name  — 人类可读名称（可选重写）
    models        — 支持的模型列表
    register_routes — 注册自己的 FastAPI 路由
    chat          — 对话接口（核心）
    check_health  — 健康检查
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from fastapi import APIRouter


class BaseBackend(ABC):
    """所有后端适配器必须实现的接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """后端唯一标识，如 'deepseek'。

        此值同时作为:
        - 路由前缀 (/v1/{name}/chat)
        - 注册表键名 (@register 用的名称)
        """
        ...

    @property
    def display_name(self) -> str:
        """人类可读的名称，用于 API 返回展示，默认同 name。"""
        return self.name

    @property
    @abstractmethod
    def models(self) -> list[str]:
        """支持的模型 ID 列表。

        客户端通过 GET /v1/{name}/models 获取此列表。
        """
        ...

    @abstractmethod
    def register_routes(self, router: APIRouter) -> None:
        """把自己的路由注册到给定的 APIRouter 上。

        一般以 /v1/{self.name} 为前缀注册 chat 和 models 两个端点。

        Args:
            router: 由 register_all_routes 分配的独立 APIRouter
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """发送对话，返回响应文本。

        Args:
            messages: OpenAI 格式的消息列表
            model: 模型 ID（来自 self.models）
            stream: True 返回异步生成器，False 返回完整文本

        Returns:
            非流式: 完整响应文本
            流式:   逐块 yield 文本的异步生成器
        """
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """检查后端是否可用（登录态是否有效）。

        Returns:
            True 表示可正常服务
        """
        ...
