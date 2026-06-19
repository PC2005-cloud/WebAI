"""DeepSeek PoW（Proof of Work）浏览器管理。

为什么需要 PoW？
----------------
DeepSeek 网页版在每次对话前需要浏览器计算一个工作量证明（PoW），
这是为了防止自动化调用。PoW 计算在浏览器中执行 JavaScript，
纯 HTTP 请求无法完成，耗时约 5-15 秒。

本模块的作用
------------
启动一个常驻的 headless Chromium，只做一件事——
拦截浏览器发往 /chat/completion 的请求，从中提取 x-ds-pow-response 头。
浏览器不参与实际的对话，对话走 HTTP 直连。

关键设计
--------
1. 浏览器跑在同步线程中（通过 asyncio.to_thread），
   通过 threading.Event 与 async 代码通信
2. 浏览器常驻，跨多次请求复用——每次 solve() 后自动导航回首页
3. 请求拦截器（route interceptor）捕获请求头后立即 abort，
   不让请求实际发出

线程安全
--------
_pow_header 和 _pow_ready 是模块级全局变量，
因为整个应用只有一个 PowSolver 实例。如果将来需要多个实例，
需要改为实例变量。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from playwright.sync_api import sync_playwright

from core.exceptions import PowTimeoutError
from gateway.deepseek.session import ANTI_DETECTION_SCRIPT, BASE_URL, SESSION_FILE

logger = logging.getLogger(__name__)

# ============================================================
# 全局 PoW 状态（跨线程通信）
#
# _pow_header: 浏览器捕获到的 x-ds-pow-response 值
# _pow_ready:  信号量，PoW 计算完成后 set()
# ============================================================

_pow_header: Optional[str] = None
_pow_ready = threading.Event()


def _make_interceptor():
    """创建路由请求拦截器。

    拦截规则:
    - 匹配 URL 中包含 "chat/completion" 的请求
    - 读取请求头中的 x-ds-pow-response
    - 捕获到后存入 _pow_header 并触发 _pow_ready
    - 中断请求（abort），不让它实际发出去

    用闭包而非类方法，因为 Playwright route handler 有函数签名要求。
    """

    def handler(route, request):
        global _pow_header
        if "chat/completion" in request.url:
            h = request.headers
            pov = h.get("x-ds-pow-response")
            if pov:
                _pow_header = pov
                _pow_ready.set()
                logger.info(
                    "[PoW] 已从浏览器捕获 PoW 响应 (len=%d)", len(pov)
                )
        route.abort()

    return handler


class PowSolver:
    """DeepSeek PoW 浏览器管理器。

    用法::
        solver = PowSolver()
        solver.start()           # 启动 headless 浏览器（~15s）
        header = solver.solve()  # 触发 PoW 计算，返回 x-ds-pow-response
        # ... 用 header 发 HTTP 请求 ...
        solver.close()           # 关闭浏览器

    注意: start() 只应调用一次，后续复用。close() 后不能再使用。
    """

    def __init__(self) -> None:
        self._pw = None
        self._browser = None
        self._page = None

    def start(self) -> None:
        """启动 headless Chromium 浏览器并打开 DeepSeek 首页。

        初始化步骤:
        1. 启动 Playwright 引擎
        2. 启动 Chromium 浏览器（headless）
        3. 创建 browser context（加载已保存的 session）
        4. 创建新页面，注入反检测脚本
        5. 注册路由拦截器，拦截 /chat/completion 请求
        6. 导航到 DeepSeek 首页
        7. 等待页面完全加载

        耗时: 首次调用约 10-15s，后续复用。
        """
        global _pow_header, _pow_ready
        _pow_header = None
        _pow_ready.clear()

        logger.info("[PoW] 启动 headless Chromium 浏览器 ...")
        t0 = time.monotonic()

        # 1-2. 启动 Playwright + Chromium
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        logger.debug("[PoW] Chromium 进程已启动")

        # 3-5. 创建 context、页面、注入脚本、注册拦截
        ctx = self._browser.new_context(
            storage_state=SESSION_FILE,
            viewport={"width": 1280, "height": 720},
        )
        self._page = ctx.new_page()
        self._page.add_init_script(ANTI_DETECTION_SCRIPT)
        self._page.route("**/chat/completion**", _make_interceptor())

        # 6-7. 加载首页
        logger.debug("[PoW] 正在加载 %s", BASE_URL)
        self._page.goto(BASE_URL, wait_until="domcontentloaded")
        self._page.wait_for_timeout(10000)

        elapsed = time.monotonic() - t0
        logger.info("[PoW] 浏览器就绪（%.1fs）", elapsed)

    def solve(self, timeout: float = 30.0) -> str:
        """触发 PoW 计算并等待结果。

        工作流程:
        1. 导航到首页，确保 textarea 可用
        2. 在 textarea 输入 "_" 并回车，触发 PoW
           （浏览器会自动发起 /chat/completion 请求，
            拦截器捕获 x-ds-pow-response 头）
        3. 等待 PoW 计算完成（最多 timeout 秒）
        4. 导航回首页，恢复页面状态供下次使用

        Args:
            timeout: 最大等待秒数（默认 30s）

        Returns:
            x-ds-pow-response header 值（字符串）

        Raises:
            PowTimeoutError: 超时未获取到 PoW
            RuntimeError: 浏览器未启动
        """
        global _pow_header, _pow_ready
        _pow_header = None
        _pow_ready.clear()

        if not self._page:
            raise RuntimeError("PoW 浏览器未启动，请先调用 start()")

        # 1. 确保页面在首页
        self._navigate_home()

        # 2. 在 textarea 输入并回车，触发 PoW
        ta = self._page.query_selector('textarea[name="search"]')
        if not ta:
            logger.debug("[PoW] textarea 未就绪，等待 5s ...")
            self._page.wait_for_timeout(5000)
            ta = self._page.query_selector('textarea[name="search"]')

        if ta:
            logger.debug("[PoW] 触发 PoW 计算 ...")
            ta.fill("_")
            self._page.wait_for_timeout(500)
            ta.press("Enter")
        else:
            logger.warning("[PoW] 未找到 textarea，尝试导航回首页重试")
            self._navigate_home()
            ta = self._page.query_selector('textarea[name="search"]')
            if ta:
                ta.fill("_")
                self._page.wait_for_timeout(500)
                ta.press("Enter")

        # 3. 等待 PoW 计算完成
        t0 = time.monotonic()
        if not _pow_ready.wait(timeout=timeout):
            elapsed = time.monotonic() - t0
            raise PowTimeoutError(
                f"PoW 计算超时（{elapsed:.0f}s / {timeout}s），"
                "DeepSeek 可能修改了验证机制"
            )

        elapsed = time.monotonic() - t0
        logger.info("[PoW] 计算完成（%.1fs）", elapsed)

        # 4. 导航回首页，下次复用
        self._navigate_home()

        return _pow_header  # type: ignore[return-value]

    def _navigate_home(self) -> None:
        """导航回 DeepSeek 首页，确保 textarea 可用。

        每次 PoW 触发后页面会进入"对话"视图，
        需要回到首页才能再次触发 PoW。
        """
        if not self._page:
            return
        try:
            self._page.goto(BASE_URL, wait_until="domcontentloaded")
            self._page.wait_for_timeout(3000)
            logger.debug("[PoW] 已导航回首页")
        except Exception as exc:
            logger.warning("[PoW] 导航回首页失败: %s", exc)

    def close(self) -> None:
        """关闭浏览器并释放所有资源。

        幂等操作，可多次调用。
        """
        logger.info("[PoW] 正在关闭浏览器 ...")

        if self._browser:
            try:
                self._browser.close()
                logger.debug("[PoW] 浏览器已关闭")
            except Exception:
                pass

        if self._pw:
            try:
                self._pw.stop()
                logger.debug("[PoW] Playwright 引擎已停止")
            except Exception:
                pass

        self._pw = None
        self._browser = None
        self._page = None
        logger.info("[PoW] 浏览器资源已释放")
