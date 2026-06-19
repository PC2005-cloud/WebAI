"""共享数据类型定义。

消息数据类，替代被诟病的 list[dict]。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# 允许的消息角色
_ROLES = ("system", "user", "assistant", "tool")


class Message(BaseModel):
    """单条消息。"""

    role: str = Field(
        description="消息角色: system / user / assistant / tool",
    )
    content: str = Field(
        default="",
        description="消息文本内容",
    )


# 消息列表类型别名，让 IDE 能透传提示
MessageList = list[Message]
