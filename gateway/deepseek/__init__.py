"""DeepSeek HTTP 直连方案模块。

本包实现了 DeepSeek 网页版的程序化调用，绕过浏览器界面，
直接与 DeepSeek 内部 HTTP API 通信。

模块结构
--------
session.py  登录态管理（token 读写、CLI 登录流程）
pow.py      PoW 工作量证明浏览器管理
sse.py      DeepSeek 非标准 SSE 流解析
client.py   _DeepSeekHTTPClient — 整合以上模块的 HTTP 客户端

依赖关系
--------
client.py → pow.py → session.py
client.py → sse.py
"""

import os

# 确保 Playwright 使用项目本地的浏览器二进制文件，
# 而不是默认的 %AppData%/ms-playwright/ 路径。
# 项目浏览器文件位于项目根目录的 browsers/ 下。
_PW_BROWSERS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "browsers")
)
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", _PW_BROWSERS_PATH)
