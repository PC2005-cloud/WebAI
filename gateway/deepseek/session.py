"""DeepSeek 登录态管理。

职责
----
1. load_token()      — 从 session JSON 文件中提取登录 Token
2. save_session()    — 将 Playwright context 的 storage_state 持久化到文件
3. is_logged_in()    — 检测浏览器页面是否已登录
4. login_cli()       — CLI 交互式登录流程

文件格式
--------
session 文件是 Playwright 的 storage_state 格式：
    {
      "cookies": [...],
      "origins": [
        {
          "origin": "https://chat.deepseek.com",
          "localStorage": [
            {"name": "userToken", "value": "...json..."},
            ...
          ]
        }
      ]
    }

Token 提取
----------
userToken 的 value 可能有两种格式:
    - 纯字符串: "abcdef12345..."
    - JSON 字符串: '{"value": "abcdef12345...", "__version": "0"}'
load_token() 兼容这两种格式。
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from playwright.sync_api import sync_playwright

from core.exceptions import SessionExpiredError

logger = logging.getLogger(__name__)

# ============================================================
# 路径常量
# ============================================================

SESSION_DIR = os.path.join("session")
SESSION_FILE = os.path.join(SESSION_DIR, "deepseek.json")
BASE_URL = "https://chat.deepseek.com"

# ============================================================
# 反检测脚本
#
# Playwright 浏览器有自动化特征（navigator.webdriver=true），
# DeepSeek 会检测到并拒绝服务。此脚本在页面加载前注入，
# 隐藏自动化痕迹。
# 参考: https://stackoverflow.com/questions/53039551/
# ============================================================

ANTI_DETECTION_SCRIPT = """
// 隐藏 webdriver 标志（最关键）
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// 填充 plugins 数组（空数组也是自动化特征）
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
        { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
        { name: 'Native Client', filename: 'pnacl' },
    ],
});

// 设置语言偏好
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });

// 模拟 chrome.runtime
window.chrome = window.chrome || {};
window.chrome.runtime = {};
"""


# ============================================================
# Token 读写
# ============================================================


def load_token(session_path: str = SESSION_FILE) -> Optional[str]:
    """从 storage_state JSON 中提取 DeepSeek 的 userToken。

    Args:
        session_path: session 文件路径，默认 session/deepseek.json

    Returns:
        Token 字符串，或 None（文件不存在/无有效 token）
    """
    if not os.path.isfile(session_path):
        logger.debug("[令牌] session 文件不存在: %s", session_path)
        return None

    try:
        with open(session_path) as f:
            data = json.load(f)

        for origin in data.get("origins", []):
            for item in origin.get("localStorage", []):
                if item.get("name") == "userToken":
                    raw = item["value"]

                    # Token 值可能是 JSON 包装的，也可能是纯字符串
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict):
                            val = parsed.get("value")
                            if val:
                                logger.debug("[令牌] 从 JSON 中提取 token 成功")
                                return val
                    except (json.JSONDecodeError, TypeError):
                        # 纯字符串格式
                        if raw:
                            logger.debug("[令牌] 提取纯文本 token 成功")
                            return raw

        logger.warning("[令牌] 未找到 userToken")
        return None

    except Exception as exc:
        logger.warning("[令牌] 读取失败: %s", exc)
        return None


def save_session(context, path: str = SESSION_FILE) -> None:
    """保存 Playwright context 的 storage_state 到文件。

    包含 cookies 和 localStorage，后续启动时可以恢复登录态。

    Args:
        context: Playwright BrowserContext 实例
        path: 保存路径，默认 session/deepseek.json
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    context.storage_state(path=path)
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    logger.info("[会话] 已保存 %s (%d 字节)", path, size)


# ============================================================
# 登录检测
# ============================================================


def is_logged_in(page) -> bool:
    """检测当前页面是否已登录。

    使用三种方法逐级检测:
    1. localStorage 中有无有效的 userToken
    2. 页面有无用户头像元素
    3. 聊天输入框是否可见且可用

    Args:
        page: Playwright Page 实例

    Returns:
        True 表示已登录
    """
    # 方法 1: localStorage token 检测
    try:
        token_raw = page.evaluate("() => localStorage.getItem('userToken')")
        if token_raw:
            # userToken 可能是 JSON {"value":"xxx"} 或纯字符串
            try:
                parsed = json.loads(token_raw)
                token_val = (
                    parsed.get("value") if isinstance(parsed, dict) else parsed
                )
            except (json.JSONDecodeError, TypeError):
                token_val = token_raw

            if token_val:
                logger.info("[认证] localStorage 检测到有效 token")
                return True
            else:
                logger.debug("[认证] userToken 键存在但值为空")
    except Exception:
        pass

    # 方法 2: 用户头像检测
    try:
        avatar = page.query_selector('img[alt*="avatar"], div[class*="avatar"]')
        if avatar and avatar.is_visible():
            logger.debug("[认证] 头像元素可见，已登录")
            return True
    except Exception:
        pass

    # 方法 3: 聊天输入框检测
    try:
        inp = page.query_selector('textarea[name="search"]')
        if inp and inp.is_visible() and inp.is_enabled():
            logger.debug("[认证] 聊天输入框可用，已登录")
            return True
    except Exception:
        pass

    return False


# ============================================================
# CLI 登录流程
# ============================================================


def login_cli(timeout_seconds: int = 120) -> None:
    """CLI 交互式登录。

    工作流程:
    1. 启动 headed Chromium（用户可见的浏览器窗口）
    2. 导航到 chat.deepseek.com
    3. 等待用户手动扫码/邮箱登录
    4. 轮询检测登录状态（每秒一次）
    5. 检测到登录后保存 session 到文件

    注意：必须在本机有 GUI 的环境运行，远程服务器无法使用。

    Args:
        timeout_seconds: 登录超时秒数（默认 120s）

    Raises:
        SessionExpiredError: 超时未完成登录
    """
    t0 = time.monotonic()

    # ── 用户提示 ──
    print()
    print("=" * 60)
    print("  ** DeepSeek 登录 **")
    print()
    print("  浏览器窗口已打开，请在浏览器中完成登录：")
    print("  支持：微信扫码 / 手机号 / 邮箱登录")
    print()
    print(f"  超时时间: {timeout_seconds} 秒")
    print("  登录完成后会话将自动保存。")
    print("=" * 60)
    print()

    # ── 启动浏览器 ──
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(
            headless=False,  # 必须显示浏览器窗口，让用户操作
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="zh-CN",
        )
        page = context.new_page()
        page.add_init_script(ANTI_DETECTION_SCRIPT)

        # 打开 DeepSeek 首页
        logger.info("[导航] 正在打开 %s", BASE_URL)
        page.goto(BASE_URL, wait_until="domcontentloaded")
        page.wait_for_timeout(5000)

        # ── 轮询等待登录 ──
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if is_logged_in(page):
                elapsed = time.monotonic() - t0
                logger.info("[登录] 检测到登录成功（%.1fs）", elapsed)
                save_session(context)
                print()
                print("=" * 60)
                print("  ** 登录成功！会话已保存至 session/deepseek.json **")
                print("=" * 60)
                print()
                return
            page.wait_for_timeout(1000)

        # ── 超时 ──
        raise SessionExpiredError(
            f"登录超时（{timeout_seconds}s），请重新运行 --mode login"
        )

    finally:
        try:
            pw.stop()
            logger.debug("[登录] Playwright 已关闭")
        except Exception:
            pass


def login_with_password(mobile: str, password: str, area_code: str = "+86") -> bool:
    """通过手机号密码登录，保存 session 文件。

    POST /api/v0/users/login
    """
    import httpx
    import uuid

    url = f"{BASE_URL}/api/v0/users/login"
    device_id = str(uuid.uuid4())

    body = {
        "email": "",
        "mobile": mobile,
        "password": password,
        "area_code": area_code,
        "device_id": device_id,
        "os": "web",
    }

    logger.info("[密码登录] 正在登录 %s%s ...", area_code, mobile[-4:])
    r = httpx.post(url, json=body, timeout=30)
    data = r.json()

    biz_code = data.get("data", {}).get("biz_code")
    if biz_code != 0:
        msg = data.get("data", {}).get("biz_msg", "未知错误")
        logger.error("[密码登录] 失败: %s", msg)
        return False

    # 提取 token（在 biz_data.user.token 中）
    biz_data = data.get("data", {}).get("biz_data", {})
    user = biz_data.get("user", {})
    token = user.get("token", "")
    if not token:
        logger.error("[密码登录] 响应中未找到 token")
        return False

    # 保存为 storage_state 格式（兼容 load_token）
    import json
    import http.cookies as http_cookies

    # 提取响应中的 cookies
    cookies_list = []
    for cookie in r.cookies.jar:
        cookies_list.append({
            "name": cookie.name,
            "value": cookie.value,
            "domain": cookie.domain or "chat.deepseek.com",
            "path": cookie.path or "/",
            "expires": int(cookie.expires) if cookie.expires else -1,
            "httpOnly": cookie.has_nonstandard_attr("httponly"),
            "secure": cookie.secure,
            "sameSite": "Lax",
        })

    storage = {
        "cookies": cookies_list,
        "origins": [
            {
                "origin": BASE_URL,
                "localStorage": [
                    {
                        "name": "userToken",
                        "value": json.dumps({"value": token, "__version": "0"}),
                    }
                ],
            }
        ],
    }
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(storage, f, indent=2)

    logger.info("[密码登录] 成功，token + %d 个 cookie 已保存", len(cookies_list))
    return True
