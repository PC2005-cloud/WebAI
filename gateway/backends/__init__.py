"""后端适配器注册机制。

通过装饰器模式实现插拔式后端注册：

    @register("deepseek")
    class DeepSeekBackend(BaseBackend):
        ...

注册表是一个全局 dict[str, type[BaseBackend]]，
在 gateway/backends/ 包导入时自动填充。

添加新后端只需三步:
    1. 在 gateway/backends/ 下创建新文件
    2. 继承 BaseBackend，用 @register("id") 装饰
    3. 在文件尾部加 import 触发注册（或让 __init__.py 自动导入）

注意: 后端模块的 import 必须在 register() 函数定义之后，
      否则出现循环导入（后端模块 import register 时它还未定义）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.exceptions import BackendNotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from fastapi import APIRouter

    from gateway.backends.base import BaseBackend

_registry: dict[str, type[BaseBackend]] = {}


def register(name: str):
    """装饰器：将后端适配器类注册到全局注册表。

    Args:
        name: 后端唯一标识，也是路由前缀（如 "deepseek"）

    用法::
        @register("my_backend")
        class MyBackend(BaseBackend):
            ...
    """

    def decorator(cls):
        _registry[name] = cls
        return cls

    return decorator


def get_backend(name: str) -> BaseBackend:
    """按名称获取后端适配器实例。

    Args:
        name: 后端唯一标识（注册时用的名称）

    Returns:
        后端适配器实例

    Raises:
        BackendNotFoundError: 名称未注册
    """
    cls = _registry.get(name)
    if not cls:
        raise BackendNotFoundError(
            f"未知后端: {name}，可用: {list(_registry.keys())}"
        )
    return cls()


def register_all_routes(router: APIRouter) -> None:
    """遍历所有已注册的后端，让它们各自注册路由。

    每个后端拥有独立的 APIRouter，最后 include_router 到主路由上。
    这样各后端的路由完全独立，互不影响。

    Args:
        router: FastAPI 主路由，各后端在其上附加自己路由
    """
    from fastapi import APIRouter

    for name, cls in _registry.items():
        backend = cls()
        backend_router = APIRouter()
        backend.register_routes(backend_router)
        router.include_router(backend_router)


def list_backends() -> list[dict]:
    """返回所有已注册后端的信息列表，供 GET /v1/backends 使用。

    Returns:
        每项包含 id, name, models, status 的列表
    """
    result = []
    for name, cls in _registry.items():
        backend = cls()
        result.append(
            {
                "id": name,
                "name": backend.display_name,
                "models": backend.models,
                "status": "ready",
            }
        )
    return result


__all__ = [
    "register",
    "get_backend",
    "register_all_routes",
    "list_backends",
]

# ============================================================
# 触发后端模块注册
#
# 这里 import 各后端模块，使它们的 @register 装饰器执行。
# 放在文件末尾，确保 register() 函数已在前面定义好。
# 添加新后端时，在这里加一行 import。
# ============================================================

import gateway.backends.deepseek  # noqa: F401, E402
import gateway.backends.doubao  # noqa: F401, E402
