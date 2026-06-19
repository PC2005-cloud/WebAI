# WebAi-Gateway — 通用 AI 网页版 → API 网关

> 将多个 AI 网页版（DeepSeek、豆包等）封装成统一的 HTTP API，供程序化调用。

---

## 1. 项目定位

**WebAi**（当前项目）是 DeepSeek 网页版 → Anthropic API 的专用适配器。  
**WebAi-Gateway**（本项目）是通用网关：**自建统一 API 格式，多后端插件化，不限 DeepSeek**。

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│  你的程序     │ ──→ │  WebAi-Gateway   │ ──→ │ DeepSeek 网页 │
│  (任何语言)   │     │                  │ ──→ │ 豆包 网页     │
│              │     │  统一 API 入口    │ ──→ │ (未来更多)    │
└──────────────┘     └──────────────────┘     └──────────────┘
```

---

## 2. 核心设计原则

1. **后端插件化** — 每个 AI 网站一个 adapter，热插拔
2. **一路由一后端** — 不同 AI 不同接口路径，不搞统一抽象
3. **HTTP 直连优先** — 尽量走内部 API，浏览器只做 PoW/反爬兜底
4. **各后端独立演进** — 一个后端挂了不影响其他

---

## 3. API 设计

每个后端有自己独立的路由，路径格式：`/v1/{backend_id}/...`

### 3.1 DeepSeek 接口

```
POST /v1/deepseek/chat
```

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "你好，用 Python 写个快排"}
  ],
  "stream": false
}
```

```
GET /v1/deepseek/models
```

```json
{
  "models": ["default", "thinking"]
}
```

### 3.2 豆包接口（示例，等调研后确定）

```
POST /v1/doubao/chat
```

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}
```

```
GET /v1/doubao/models
```

### 3.3 通用格式约定

每个后端的 chat 接口**遵循相同的响应结构**，方便客户端切换后端时少改代码：

**非流式响应：**

```json
{
  "id": "chat-xxx",
  "content": "以下是快排实现...",
  "finish_reason": "stop"
}
```

**流式响应（SSE）：**

```
data: {"type": "content", "content": "以下"}
data: {"type": "content", "content": "是快排"}
data: {"type": "done", "finish_reason": "stop"}
```

### 3.4 后端列表（元信息）

```
GET /v1/backends
```

```json
{
  "backends": [
    {"id": "deepseek", "name": "DeepSeek", "models": ["default", "thinking"], "status": "ready"},
    {"id": "doubao", "name": "豆包", "models": ["default"], "status": "coming_soon"}
  ]
}
```

---

## 4. 项目结构

```
webai-gateway/
├── pyproject.toml
├── README.md
├── gateway/
│   ├── __init__.py
│   ├── server.py            # FastAPI 主服务，组装各后端路由
│   └── backends/            # ---- 后端适配器目录 ----
│       ├── __init__.py
│       ├── base.py          # BaseBackend 抽象类
│       ├── deepseek.py      # DeepSeek adapter（自己注册路由）
│       ├── doubao.py        # 豆包 adapter（自己注册路由）
│       └── ...
├── core/
│   ├── __init__.py
│   ├── browser.py           # 共享的 Playwright 浏览器管理
│   └── exceptions.py        # 异常体系
└── session/
    ├── deepseek.json        # DeepSeek 的登录态
    └── doubao.json          # 豆包的登录态
```

---

## 5. Backend Adapter 接口

每个后端适配器继承 `BaseBackend`，除了实现对话逻辑，还要**注册自己的路由**：

```python
from abc import ABC, abstractmethod
from typing import AsyncGenerator
from fastapi import APIRouter

class BaseBackend(ABC):
    """所有后端适配器必须实现的接口"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str = "default",
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """发送对话，返回文本或流式生成器"""
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """检查后端是否可用（登录态是否有效）"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """后端唯一标识，如 'deepseek'，也是路由前缀"""
        ...

    @property
    @abstractmethod
    def models(self) -> list[str]:
        """支持的模型列表"""
        ...

    @abstractmethod
    def register_routes(self, router: APIRouter):
        """把自己的路由注册到全局 router 上
        
        每个后端用 self.name 作为前缀，如 /v1/deepseek/chat
        """
        ...
```

### Backend 注册

```python
# gateway/backends/__init__.py
_registry: dict[str, type[BaseBackend]] = {}

def register(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls
    return decorator

def get_backend(name: str) -> BaseBackend:
    cls = _registry.get(name)
    if not cls:
        raise ValueError(f"未知后端: {name}，可用: {list(_registry.keys())}")
    return cls()

def register_all_routes(router: APIRouter):
    """遍历所有已注册的后端，让它们各自注册路由"""
    for name, cls in _registry.items():
        backend = cls()
        backend.register_routes(router)
```

添加一个新后端只需要三步：

```python
@register("doubao")
class DoubaoBackend(BaseBackend):
    @property
    def name(self):
        return "doubao"

    @property
    def models(self):
        return ["default"]

    def register_routes(self, router: APIRouter):
        prefix = f"/v1/{self.name}"
        # 每个后端在自己内部定义路由，自由度最高
        router.add_api_route(
            f"{prefix}/chat", self.chat_endpoint, methods=["POST"]
        )
        router.add_api_route(
            f"{prefix}/models", self.models_endpoint, methods=["GET"]
        )

    async def chat(self, messages, model="default", stream=False):
        # 实现豆包网页版的调用逻辑
        ...

    async def chat_endpoint(self, request: Request):
        # 解析请求体 → 调 self.chat() → 包装响应
        ...

    async def check_health(self):
        # 检查登录态是否有效
        ...
```

---

## 6. 后端实现策略

### 6.1 DeepSeek Adapter

直接从当前 WebAi 项目移植 `deepseek_api.py` 的逻辑：

- HTTP 直连 DeepSeek 内部 API
- 共享的 headless 浏览器做 PoW
- 从 `session/storage_state.json` 读取登录态（DeepSeek 的 key 下）

### 6.2 豆包 Adapter

豆包网页版需要调研它的内部 API 和反爬机制：

- 可能方案 A：找到豆包的内部 HTTP API，类似 DeepSeek 的做法
- 可能方案 B：Playwright 全浏览器自动化（如果反爬严格）
- 登录态统一管理，在 `storage_state.json` 中用 origin 区分

### 6.3 浏览器资源管理

多个后端如果都需要浏览器（PoW、反爬），**共享一个浏览器池**：

```python
# core/browser.py
class BrowserPool:
    """全局共享的 Playwright 浏览器池"""
    
    async def get_page(self, backend: str) -> Page:
        """为某后端获取一个页面（复用或新建）"""
        ...
    
    async def release_page(self, backend: str, page: Page):
        """释放页面回池"""
        ...
```

---

## 7. 与 WebAi 项目的关键区别

| 维度 | WebAi（当前） | WebAi-Gateway（新） |
|------|-------------|---------------------|
| API 格式 | Anthropic Messages API 兼容 | **自研统一 API** |
| 后端数量 | 仅 DeepSeek | **多后端插件化** |
| 协议转换 | Anthropic ↔ DeepSeek | 统一格式 ↔ 各后端 |
| 工具调用 | 有（tool_use 兼容） | **先不做**，v1 只做纯对话 |
| Claude Code 兼容 | 是 | 否 |
| 架构复杂度 | 单体 | **模块化 + 插件** |

---

## 8. 分阶段实施计划

### Phase 1：骨架（预计 1 个周末）
- [ ] 项目初始化，`pyproject.toml`，基础目录结构
- [ ] `BaseBackend` 抽象类定义
- [ ] `server.py` + FastAPI 端点骨架（/v1/chat, /v1/backends）
- [ ] `schema.py` 请求/响应模型
- [ ] `backends/__init__.py` 注册机制
- [ ] 能启动服务，返回 501（未实现）

### Phase 2：DeepSeek Adapter（预计 1-2 天）
- [ ] 从 WebAi 移植 HTTP 直连逻辑
- [ ] 适配 `BaseBackend` 接口
- [ ] 配置 / 登录态管理
- [ ] 测试通过

### Phase 3：豆包 Adapter（预计 2-3 天）
- [ ] 调研豆包网页版的 API / 反爬机制
- [ ] 实现 `DoubaoBackend`
- [ ] 测试通过

### Phase 4：浏览器池 & 稳定性（按需）
- [ ] `BrowserPool` 实现
- [ ] 错误重试
- [ ] 登录态过期检测与提醒
- [ ] 日志完善

### Phase 5：增强（可选）
- [ ] 工具调用支持（参考 WebAi 的 `<tool_call>` 方案）
- [ ] Docker 镜像
- [ ] 配置文件（YAML/TOML）管理多后端参数

---

## 9. 目录创建指令

在你开新 AI 智能体时，先执行：

```bash
mkdir -p d:/Projects/Python_word/webai/gateway/backends
mkdir -p d:/Projects/Python_word/webai/core
mkdir -p d:/Projects/Python_word/webai/session
```

然后按 Phase 1 → Phase 2 的顺序逐步实现。

---

## 10. 难点与实现细节（必读！）

> ⚠️ 以下内容是现有 WebAi 项目踩过的坑，新 agent 实现前**先通读一遍**，避免重新造轮子或踩同样的坑。

---

### 10.1 DeepSeek PoW（工作量证明）—— 最大难点

**问题**：DeepSeek 每次对话前要求计算 Proof of Work，约 5-15 秒。PoW 必须在浏览器环境执行，纯 HTTP 请求无法完成。

**现有方案（已验证可工作）**：

启动一个 **headless Chromium** 常驻后台，不参与对话，只干一件事——拦截 PoW 响应头。

```python
# 核心逻辑（来自 WebAi 的 _HTTPWorker）
pw = sync_playwright().start()
browser = pw.chromium.launch(headless=True)
ctx = browser.new_context(storage_state=session_file)
page = ctx.new_page()

pow_header = [None]
pow_ready = threading.Event()

def interceptor(route, req):
    if "chat/completion" in req.url:
        p = req.headers.get("x-ds-pow-response")
        if p:
            pow_header[0] = p
            pow_ready.set()
    route.abort()

page.route("**/chat/completion**", interceptor)
page.goto("https://chat.deepseek.com")
```

**触发 PoW 计算**：在页面 textarea 输入任意字符并回车，浏览器自动触发 PoW，拦截器捕获 `x-ds-pow-response`。

```python
ta = page.query_selector('textarea[name="search"]')
ta.fill("_")
ta.press("Enter")
pow_ready.wait(timeout=30)  # 最多等 30s
```

**⚠️ 坑**：
- PoW 浏览器启动后需要等页面完全加载（约 10s）
- `pow_ready` 是 `threading.Event`，每次对话前要 `clear()` 重置
- 如果 PoW 超时（30s），说明 DeepSeek 可能改了算法或页面结构变了
- PoW 浏览器在请求完成后不要关，保持常驻，下次复用

**⚠️ 线程安全**：
- PoW 浏览器跑在**独立线程**里（`_HTTPWorker`），不是 async
- 和 FastAPI 的 async 事件循环之间通过 `queue.Queue` 通信
- `server.py` 的做法：`_http_sess.post(...)` 是同步的，放在工作线程里调，不会阻塞 FastAPI

---

### 10.2 DeepSeek SSE 响应解析（非标准格式）

**问题**：DeepSeek 的 completion API 返回的是 SSE 流，但格式**不是标准 OpenAI SSE**。它用 `p`（路径）、`o`（操作）、`v`（值）字段描述增量更新。

```json
// 来自 DeepSeek SSE 的 data: 行示例
data: {"p": "response/content", "o": "APPEND", "v": "以下是"}
data: {"p": "response/content", "o": "APPEND", "v": "快排实现"}
data: {"p": "response/thinking", "o": "APPEND", "v": "思考过程..."}
```

**解析逻辑（已验证）**：

```python
chunks = []
current_path = ""
for line in resp_text.split("\n"):
    if not line or not line.startswith("data: "):
        continue
    s = line[6:].strip()
    if not s:
        continue
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        continue
    if d.get("p"):
        current_path = d["p"]
    v = d.get("v")
    o = d.get("o")
    # 只收集 response/content 路径下的 APPEND 操作
    if current_path == "response/content" and isinstance(v, str):
        if o == "APPEND" or not o:
            chunks.append(v)
return "".join(chunks).strip()
```

**⚠️ 关键点**：
- `p` 字段是**路径继承**的——不是每行都带，缺省时沿用上一行的 path
- `response/content` 是用户看到的回答，`response/thinking` 是思维链（思考过程）
- 非流式请求（`stream=False`）也会返回 SSE 格式的完整 body，直接 `resp.text` 拿全部再逐行解析
- 如果 `resp.text` 为空，检查 `stream=False` 是否生效——有些版本的 requests 库会因编码问题截断响应

---

### 10.3 登录态管理（storage_state.json）

**问题**：Playwright 的 `storage_state` 包含 cookies 和 localStorage，不同后端的登录态混在一起时会冲突。

**现有做法**：

```python
# 登录：打开浏览器，用户手动扫码/邮箱登录，保存会话
# 文件 session/storage_state.json
{
  "cookies": [...],
  "origins": [
    {
      "origin": "https://chat.deepseek.com",
      "localStorage": [
        {"name": "userToken", "value": "..."},
        {"name": "some_key", "value": "..."}
      ]
    }
  ]
}
```

**读取 userToken**（DeepSeek 的 token 藏在 localStorage 里）：

```python
def load_token():
    with open("session/storage_state.json") as f:
        data = json.load(f)
    for origin in data.get("origins", []):
        for item in origin.get("localStorage", []):
            if item.get("name") == "userToken":
                raw = item.get("value", "")
                return json.loads(raw).get("value") or raw
    return None
```

**⚠️ 多后端共存方案**（重要）：

新项目有多个后端时，**不要用一个 storage_state.json 混在一起**。建议方案：

```
session/
├── deepseek.json        # DeepSeek 的 cookies + localStorage
├── doubao.json          # 豆包的 cookies + localStorage
└── storage_state.json   # （弃用，保持兼容）
```

**token 过期检测**：
- DeepSeek token 有效期不确定，过期后 API 返回 401
- `check_health()` 定期发一个轻量请求验证 token 是否有效
- 过期时返回 402 错误码并提示用户重新 `--mode login`

---

### 10.4 DeepSeek 内部 API 端点（已验证）

**注意**：这些是 DeepSeek 的内部 API，随时可能变更。

```python
_DS_BASE = "https://chat.deepseek.com"
_DS_CHALLENGE = f"{_DS_BASE}/api/v0/chat/create_pow_challenge"
_DS_SESSION = f"{_DS_BASE}/api/v0/chat_session/create"
_DS_COMPLETION = f"{_DS_BASE}/api/v0/chat/completion"
```

**调用顺序**：

```
1. 获取 session：POST /api/v0/chat_session/create  → 得到 session_id
2. 触发 PoW：浏览器操作 → 得到 x-ds-pow-response header
3. 发对话：POST /api/v0/chat/completion → SSE 响应
```

**Session 请求体**：`{}`（空对象）

**Session 响应解析**：

```python
r = requests.post(_DS_SESSION, json={}, headers={"Authorization": f"Bearer {token}"})
j = r.json()
data = j.get("data")
sid = data["biz_data"]["id"]  # session_id
```

**Completion 请求体**：

```python
body = {
    "chat_session_id": sid,
    "parent_message_id": None,
    "model_type": "default",     # "default" 或 "thinking"
    "prompt": prompt_text,
    "ref_file_ids": [],
    "thinking_enabled": True,
    "search_enabled": True,
    "action": None,
    "preempt": False,
}
headers = {
    "Authorization": f"Bearer {token}",
    "X-Ds-Pow-Response": pow_header_value,
}
resp = requests.post(_DS_COMPLETION, json=body, stream=False, timeout=300, headers=headers)
```

**⚠️ 要点**：
- `model_type` 参数：`"default"` 是普通模型，`"thinking"` 是深度思考（R1）
- `thinking_enabled` 设为 `True` 会返回思维链内容（在 `response/thinking` 路径下）
- `parent_message_id` 设为 `None` 是单轮对话，多轮对话需要传上一轮的消息 ID
- 多轮对话的 prompt 需要**包含历史消息**（见 10.6）

---

### 10.5 浏览器反检测（关键）

Playwright 打开的浏览器有自动化特征，DeepSeek（和其他 AI 网站）能检测到。

**必需的反检测措施**：

```python
# 1. 禁用 AutomationControlled 特性
LAUNCH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
]

# 2. 注入反检测脚本（在 page.goto 之前执行）
page.add_init_script("""
    // 隐藏 webdriver 标志
    Object.defineProperty(navigator, 'webdriver', { get: () => false });
    
    // 填充 plugins（空数组也是自动化特征）
    Object.defineProperty(navigator, 'plugins', { 
        get: () => [1, 2, 3, 4, 5] 
    });
    
    // 覆盖 chrome 属性
    window.chrome = { runtime: {} };
""")
```

**⚠️ 坑**：
- `add_init_script` 必须在 `page.goto()` **之前**调用
- 每个新页面都要重新执行一次反检测脚本
- 不同网站对自动化检测的严格程度不同（豆包可能比 DeepSeek 更严）
- 保持 Playwright 版本更新，旧版本的特征更容易被检测

---

### 10.6 提示词组装（messages → text）

**问题**：网页版 AI 接受纯文本输入，需要把 OpenAI 格式的 messages 数组拼接成一段连续的文本。

```python
def build_prompt(messages: list[dict]) -> str:
    """将 messages 数组转为纯文本提示词"""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"[系统指令]\n{content}")
        elif role == "user":
            parts.append(f"[用户]\n{content}")
        elif role == "assistant":
            parts.append(f"[助手]\n{content}")
        elif role == "tool":
            parts.append(f"[工具执行结果]\n{content}")
    return "\n\n".join(parts)
```

**⚠️ role 标签命名**：AI 网页版没有经过指令微调来识别 `[系统]`/`[用户]` 这样的标签，所以标签名不重要，重要的是**内容和顺序的连贯性**。可以用任何分隔方式。

**多轮对话**：messages 数组天然包含历史消息，直接全部拼接即可。没有 `parent_message_id` 的烦恼。

---

### 10.7 豆包 Adapter 实现注意事项

豆包目前还没有被实现在现有项目中，以下**推测**基于通用经验：

**调研方向**：
1. 打开豆包网页版 → F12 → Network → 发一条消息 → 观察请求
2. 寻找 `chat/completion`、`conversation` 等关键词
3. 看请求头是否有 `Authorization`、`x-*` 自定义头
4. 看响应是 SSE 流还是 JSON

**可能的反爬措施**：
- 可能有类似 PoW 的验证机制
- 可能需要 WebSocket 而非 HTTP
- 可能需要特定的 header 或 cookie
- 可能检测 `navigator.webdriver` 等 Playwright 特征

**通用调试技巧**：

```python
# 1. 用 Playwright 打开页面，手动登录后抓包
page = browser.new_page()
page.goto("https://www.doubao.com")
# 手动登录... 然后观察 Network 请求

# 2. 记录所有请求和响应
page.on("request", lambda req: print(f"→ {req.method} {req.url}"))
page.on("response", lambda resp: print(f"← {resp.status} {resp.url}"))

# 3. 保存请求体的副本
page.on("request", lambda req: 
    print(f"BODY: {req.post_data[:500]}" if req.post_data else ""))
```

**如果方案 A（HTTP 直连）走不通**，回退到方案 B（全浏览器自动化）：
- 用 Playwright 在文本框中打字 → 点击发送 → 等回复 → 提取内容
- 这个方案慢但不依赖内部 API，只要网站界面不变就能用

---

### 10.8 浏览器池设计要点

多个后端可能共享同一个 Playwright 浏览器实例，但需要注意：

```python
class BrowserPool:
    """Playwright 多后端共享"""
    
    def __init__(self):
        self._pw = None
        self._browser = None
        self._pages: dict[str, list[Page]] = {}  # backend → pages
    
    def start(self):
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
    
    def get_page(self, backend: str, storage_state: str) -> Page:
        """为某后端创建一个新页面"""
        ctx = self._browser.new_context(
            storage_state=storage_state,
            viewport={"width": 1280, "height": 720},
        )
        page = ctx.new_page()
        # 注入反检测脚本
        page.add_init_script("...")
        return page
    
    def shutdown(self):
        try: self._browser.close()
        except: pass
        try: self._pw.stop()
        except: pass
```

**⚠️ 关键点**：
- **每个后端一个独立的 `browser_context`**，不要共享 context，因为每个后端的 cookie/localStorage 不同
- 每个 context 可以复用一个 `browser` 实例（Chromium 进程共享，轻量）
- PoW 浏览器是**特殊的单例**——整个服务只需要一个页面专门做 PoW，不要为每个请求都创建

---

### 10.9 超时与错误处理

参考 WebAi 的经验，建议使用以下超时配置：

| 超时类型 | 推荐值 | 说明 |
|---------|--------|------|
| HTTP 请求 | 300s | AI 长回答可能需要几分钟 |
| PoW 等待 | 30s | PoW 一般 5-15s，30s 是安全上限 |
| SSE 解析 | 无 | 等 HTTP 请求结束再解析，不单独设超时 |
| 队列等待 | 300s | 工作线程忙时排队等待 |

**异常体系**：

```python
class WebAiError(Exception): pass
class SessionExpiredError(WebAiError): pass    # token 过期
class BackendNotAvailableError(WebAiError): pass  # 后端不可用
class PowTimeoutError(WebAiError): pass        # PoW 超时
class ResponseTimeoutError(WebAiError): pass   # AI 响应超时
```

---

### 10.10 关于 `sync_playwright` vs `async_playwright`

现有 WebAi 用的是 `sync_playwright`（同步 API）+ 工作线程，因为 PoW 浏览器需要常驻在独立线程里。

**建议新项目也先用同步版**，原因：
- PoW 捕获（`interceptor`）是同步的 / callback 风格的
- 工作线程 + `queue.Queue` 的通信模型简单可靠
- 和 FastAPI async 不冲突（工作线程跑同步代码，FastAPI 跑 async）

如果用 `async_playwright`，需要注意 `interceptor` 的异步回调处理，复杂度更高。**不建议 v1 就用 async_playwright**。
