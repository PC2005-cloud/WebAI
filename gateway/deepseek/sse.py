"""DeepSeek 非标准 SSE 流解析器。

背景
----
DeepSeek 的 /api/v0/chat/completion 返回的是 SSE 流，
但格式与 OpenAI 标准 SSE 不同。它使用三个字段描述增量更新：

    data: {"p": "response/content", "o": "APPEND", "v": "这是内容"}

字段说明:
    p (path):   内容路径，决定这段数据属于哪个字段
    o (op):     操作类型（APPEND / SET / 省略）
    v (value):  值

路径继承规则
------------
p 字段是**路径继承**的——不是每行都带，缺省时沿用上一行的 path：

    data: {"p": "response/content", "o": "APPEND", "v": "你好"}  ← p = "response/content"
    data: {"v": "世界"}                                              ← p 沿用 "response/content"

路径列表
--------
| 路径                          | 含义           | 操作     |
|-------------------------------|---------------|----------|
| response/content              | 对话内容       | APPEND   |
| response/thinking_content     | 思维链内容     | APPEND   |
| response/thinking_elapsed_secs| 思维链耗时     | SET      |
| response/accumulated_token_usage | token 用量  | SET      |
| response/status               | 状态           | SET      |
| response/fragments            | 片段           | APPEND   |

注意: 早期 API 使用 "response/thinking" 路径，
      新版已改为 "response/thinking_content"。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


def parse_sse_line(line: str) -> Optional[dict[str, Any]]:
    """解析单行 SSE data。

    Args:
        line: 原始行文本
              例: 'data: {"p": "response/content", "v": "你好"}'

    Returns:
        解析后的 dict，或 None（跳过空行/非 data 行/解析失败行）
    """
    if not line or not line.startswith("data: "):
        return None

    s = line[6:].strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        logger.debug("[SSE] JSON 解析失败: %s", line[:80])
        return None


def extract_content_from_sse(sse_text: str) -> str:
    """从完整的 SSE 响应体中提取对话内容文本。

    只收集 response/content 路径下的 APPEND 操作，
    忽略 thinking_content、status 等其他字段。

    Args:
        sse_text: SSE 完整响应文本（多行 data: 条目）

    Returns:
        拼接后的纯文本内容
    """
    chunks: list[str] = []
    last_p = ""

    for line in sse_text.split("\n"):
        d = parse_sse_line(line)
        if d is None:
            continue

        p = d.get("p") or last_p
        v = d.get("v")
        o = d.get("o")
        if p:
            last_p = p

        if p == "response/content" and isinstance(v, str):
            if o == "APPEND" or not o:
                chunks.append(v)

    result = "".join(chunks).strip()
    logger.debug(
        "[SSE] 从 %d 行中提取了 %d 字符内容",
        len(sse_text.split("\n")),
        len(result),
    )
    return result


def extract_thinking_from_sse(sse_text: str) -> str:
    """从 SSE 响应体中提取思维链内容（response/thinking_content）。

    DeepSeek 的深度思考模式会输出思维链内容，
    路径为 "response/thinking_content"（新版API）。

    Args:
        sse_text: SSE 完整响应文本

    Returns:
        思维链文本
    """
    chunks: list[str] = []
    last_p = ""

    for line in sse_text.split("\n"):
        d = parse_sse_line(line)
        if d is None:
            continue

        p = d.get("p") or last_p
        v = d.get("v")
        o = d.get("o")
        if p:
            last_p = p

        if p == "response/thinking_content" and isinstance(v, str):
            if o == "APPEND" or not o:
                chunks.append(v)

    return "".join(chunks).strip()


def stream_content_from_sse(lines: list[str]) -> Generator[str, None, None]:
    """流式模式：逐行解析 SSE 并实时 yield 内容块。

    与 extract_content_from_sse 的区别：
    前者等全部收集完再返回，这个逐条 yield，适合流式 SSE 传输。

    Args:
        lines: SSE 文本行列表（可通过 resp.iter_lines() 获得）

    Yields:
        每段实时内容文本块
    """
    last_p = ""
    for line in lines:
        d = parse_sse_line(line)
        if d is None:
            continue

        p = d.get("p") or last_p
        v = d.get("v")
        o = d.get("o")
        if p:
            last_p = p

        if p == "response/content" and isinstance(v, str) and v:
            yield v
