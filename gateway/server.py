"""WebAi-Gateway — FastAPI 主服务。

架构概要
--------
本服务是统一 API 入口，采用插件化后端架构：

    HTTP 请求 → FastAPI 路由 → 后端适配器 (如 DeepSeekBackend)
                                  ├── 登录态管理 (session.py)
                                  ├── PoW 浏览器 (pow.py)
                                  ├── HTTP 直连客户端 (client.py)
                                  └── SSE 解析器 (sse.py)

每个后端适配器在 gateway/backends/ 中独立实现，
通过 @register 装饰器自动注册路由。
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from core.exceptions import (
    BackendNotAvailableError,
    BackendNotFoundError,
    PowTimeoutError,
    SessionExpiredError,
    WebAiError,
)
from core.response import Result
from gateway.backends import list_backends, register_all_routes

logger = logging.getLogger(__name__)

# 模块加载时即配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 抑制 httpx 和 uvicorn.access 的 INFO 日志（太多噪音）
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)

# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="WebAi-Gateway",
    description=(
        "将多个 AI 网页版（DeepSeek、豆包等）封装成统一的 HTTP API。"
        "每个后端有独立的路由前缀，如 /v1/deepseek/chat。\n\n"
        "## 响应格式\n"
        "所有接口返回统一 Result 结构：\n"
        "```json\n"
        '{"code": 1, "message": "success", "data": {...}}\n'
        "```\n"
        "## 后端列表\n"
        "- **DeepSeek** — HTTP 直连，支持流式/非流式对话\n"
        "- **豆包** — 待实现"
    ),
    version="0.1.0",
    docs_url="/docs",
)


# ============================================================
# 全局异常处理器
# ============================================================


@app.exception_handler(WebAiError)
async def webai_error_handler(request, exc: WebAiError):
    """捕获 WebAiError 异常，返回 Result 格式。"""
    status_map = {
        SessionExpiredError: 402,
        PowTimeoutError: 503,
        BackendNotAvailableError: 502,
        BackendNotFoundError: 404,
    }
    status = 500
    for exc_type, code in status_map.items():
        if isinstance(exc, exc_type):
            status = code
            break
    logger.warning("[%d] %s: %s", status, type(exc).__name__, exc)
    return JSONResponse(status_code=status, content=Result.error(str(exc)).model_dump())


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """捕获参数校验错误，返回 400。"""
    logger.warning("[400] ValueError: %s", exc)
    return JSONResponse(status_code=400, content=Result.error(str(exc)).model_dump())


# ============================================================
# 全局端点
# ============================================================


@app.get(
    "/health",
    summary="健康检查",
    description="返回服务运行状态，用于负载均衡和监控探针。",
    tags=["系统"],
    response_model=Result,
)
async def health():
    """服务健康检查。"""
    return Result.success({"status": "ok"})


@app.get(
    "/v1/backends",
    summary="列出所有后端",
    description="返回所有已注册的 AI 后端适配器，包含名称、支持模型和运行状态。",
    tags=["系统"],
    response_model=Result,
)
async def v1_backends():
    """列出所有已注册的后端。"""
    backends = list_backends()
    logger.info("GET /v1/backends → %d 个后端", len(backends))
    return Result.success(backends)


# ============================================================
# 注册后端路由（各后端通过 @register 自动注册，见 backends/__init__.py）
# ============================================================

register_all_routes(app.router)


# ============================================================
# CLI 入口
# ============================================================


def cmd_login() -> None:
    """CLI 模式：运行 DeepSeek 登录流程。"""
    from gateway.deepseek.session import login_cli

    logger.info("进入登录模式")
    login_cli()


def cmd_serve(args: argparse.Namespace) -> None:
    """CLI 模式：启动 FastAPI HTTP 服务。"""

    backends = [b["id"] for b in list_backends()]
    logger.info("=" * 50)
    logger.info("  WebAi-Gateway v0.1.0 启动")
    logger.info("  监听地址: %s:%d", args.host, args.port)
    logger.info("  已注册后端: %s", ", ".join(backends) if backends else "(无)")
    logger.info("  热重载: %s", "开" if args.reload else "关")
    logger.info("=" * 50)

    uvicorn.run(
        "gateway.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_excludes=["**/__pycache__/**", "**/*.pyc", "**/*.log", "**/session/**"],
        log_config=None,
    )


def main() -> None:
    """CLI 主入口。

    支持两种模式:
        --mode serve  启动 HTTP 服务（默认）
        --mode login  执行 DeepSeek 登录
    """
    parser = argparse.ArgumentParser(
        description="WebAi-Gateway — 通用 AI 网页版 API 网关"
    )
    parser.add_argument(
        "--mode",
        choices=["serve", "login"],
        default="serve",
        help="运行模式（默认: serve）",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="监听地址（默认: 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="监听端口（默认: 8888）",
    )
    parser.add_argument(
        "--no-reload",
        action="store_false",
        dest="reload",
        help="禁用热重载",
    )
    parser.set_defaults(reload=True)

    args = parser.parse_args()
    logger.debug("CLI 参数: %s", args)

    if args.mode == "login":
        cmd_login()
    else:
        cmd_serve(args)


if __name__ == "__main__":
    main()
