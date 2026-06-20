"""DeepSeek 新版 SSE 流解析器。

版 SSE 使用 `response/fragments/-1/content` 路径推送内容增量。

    data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "这是内容"}

字段说明:
    p (path):   内容路径
    o (op):     操作类型（APPEND / SET / BATCH）
    v (value):  值

路径继承规则
------------
p 字段是**路径继承**的——缺省时沿用上一行的 path：

    data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "你好"}
    data: {"v": "世界"}  ← p 沿用上一行

路径列表
--------
| 路径                             | 含义         | 操作     |
|----------------------------------|-------------|----------|
| response/fragments/-1/content    | 对话内容     | APPEND   |
| response/thinking_content        | 思维链内容   | APPEND   |
| response/fragments/-1/elapsed_secs | 片段耗时   | SET      |
| response/accumulated_token_usage | token 用量   | SET      |
| response/status                  | 状态         | SET      |
"""

from __future__ import annotations

import json
import logging
import time
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
    """从 SSE 响应体中提取对话内容。"""
    t0 = time.monotonic()
    chunks: list[str] = []
    last_p = ""
    lines = 0

    for line in sse_text.split("\n"):
        d = parse_sse_line(line)
        if d is None:
            continue
        lines += 1
        p = d.get("p") or last_p
        v = d.get("v")
        o = d.get("o")
        if p:
            last_p = p

        if p == "response/fragments/-1/content" and isinstance(v, str):
            if o == "APPEND" or not o:
                chunks.append(v)

    result = "".join(chunks).strip()
    logger.debug("[SSE] 解析 %d 行 → %d 字符 (%.2fs)", lines, len(result), time.monotonic() - t0)
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

        if p == "response/fragments/-1/content" and isinstance(v, str) and v:
            yield v
