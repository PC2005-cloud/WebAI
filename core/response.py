"""统一 API 响应格式。

所有接口统一返回 Result 结构，方便客户端统一处理：

    {
      "code": 1,           # 1=成功，-1=失败
      "message": "success",
      "data": { ... }      # 业务数据
    }

异常情况直接抛 HTTPException，由 FastAPI 异常处理器捕获后返回。
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

_DATA = TypeVar("_DATA")


class Result(BaseModel, Generic[_DATA]):
    """统一 API 响应体。"""

    code: int = Field(description="状态码，1=成功，-1=失败")
    message: str = Field(description="提示信息")
    data: _DATA | None = Field(None, description="业务数据，失败时为 null")

    @staticmethod
    def success(data: Any = None) -> "Result":
        """创建成功响应。

        Args:
            data: 业务数据（可选）

        Returns:
            Result(code=1, message="success", data=data)
        """
        return Result(code=1, message="success", data=data)

    @staticmethod
    def error(message: str, data: Any = None) -> "Result":
        """创建失败响应。

        Args:
            message: 错误描述
            data: 附加信息（可选）

        Returns:
            Result(code=-1, message=message, data=data)
        """
        return Result(code=-1, message=message, data=data)
