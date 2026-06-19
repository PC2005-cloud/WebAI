# WebAi-Gateway

将 DeepSeek 网页版封装成 HTTP API，让你可以用 curl 调 AI。

## 快速开始

```bash
# 1. 装依赖
uv sync

# 2. 登录（会弹出浏览器，扫码或输账号）
uv run python -m gateway.server --mode login

# 3. 启动服务
uv run python -m gateway.server
```

## 调用

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/backends

curl -X POST http://localhost:8000/v1/deepseek/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"model":"default"}'
```

沉浸式看接口文档：`http://localhost:8000/docs`

## 项目状态

- DeepSeek 基础对话 ✅
- 豆包、更多 AI 后端 ❌（待实现）
