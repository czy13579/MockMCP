# Mock MCP 和 Mock FastMCP 使用教程

`MockMcp` 和 `MockFastMcp` 类提供了一种在不进行实际网络通信（如stdio或SSE）的情况下，直接调用 `mcp.server.lowlevel.server.Server` 和 `fastmcp.server.server.FastMCP` 实例中定义的处理器（工具、资源、提示）的方法。这对于单元测试、集成测试和本地开发非常有用。

## MockMcp (用于 `mcp.server.lowlevel.server.Server`)

`MockMcp` 模拟了底层的 `Server`。

### 1. 创建和配置 `Server` 实例

首先，像往常一样创建和配置你的 `mcp.server.lowlevel.server.Server` 实例：

```python
# server_setup.py
import mcp.types as types
from mcp.server.lowlevel.server import Server, LifespanResultT
from contextlib import asynccontextmanager
from typing import AsyncIterator

# 可选：定义 lifespan 上下文
class MyLifespanCtx:
    db_connection: str = "dummy_connection"

@asynccontextmanager
async def my_lifespan(server: Server[MyLifespanCtx]) -> AsyncIterator[MyLifespanCtx]:
    print("Lifespan init")
    ctx = MyLifespanCtx()
    yield ctx
    print("Lifespan shutdown")

server = Server[MyLifespanCtx]("my_server", lifespan=my_lifespan)

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name="my_tool", inputSchema={"type": "object"})]

@server.call_tool()
async def call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if name == "my_tool":
        # 可以尝试访问 context (如果处理器需要)
        # from mcp.server.lowlevel.server import request_ctx
        # ctx = request_ctx.get()
        # lifespan_ctx = ctx.lifespan_context # 类型为 MyLifespanCtx
        # print(f"Accessing lifespan context: {lifespan_ctx.db_connection}")
        arg_val = arguments.get("data", "default") if arguments else "default"
        return [types.TextContent(type="text", text=f"Tool result: {arg_val}")]
    raise ValueError(f"Unknown tool: {name}")

# ... 其他处理器 ...
```

### 2. 将 `Server` 映射到 `MockMcp`

使用 `map_server_to_mock` 异步函数创建 `MockMcp` 实例。这会复制处理器并执行（但不关闭）服务器的 lifespan 来获取上下文。

```python
import asyncio
# 假设 mock_mcp.py 在 python 路径中
from mock_mcp import map_server_to_mock, MockMcp
# 从你的设置文件中导入 server 实例
from server_setup import server, MyLifespanCtx

async def run_mock():
    # 映射 Server 到 MockMcp
    # 指定类型参数 [MyLifespanCtx] 很重要
    mock_server: MockMcp[MyLifespanCtx] = await map_server_to_mock(server)

    print(f"Mock Server Name: {mock_server.name}")
    # 检查 lifespan 上下文是否被捕获
    print(f"Lifespan context status: {mock_server._lifespan_context.db_connection}")

    # ... 调用 mock server 的方法 ...

# if __name__ == "__main__":
#     asyncio.run(run_mock())
```

### 3. 直接调用 Mock 方法

现在你可以直接调用 `MockMcp` 实例上的 `async` 方法，它们会模拟 MCP 请求并返回解包后的结果。

```python
# (在 run_mock 函数内部)
    try:
        print("\nCalling list_tools...")
        tools = await mock_server.list_tools()
        print(f"Tools: {tools}")

        print("\nCalling my_tool...")
        result_content = await mock_server.call_tool("my_tool", {"data": "test data"})
        # call_tool 返回 content 列表
        print(f"Result: {result_content[0].text}")

        print("\nCalling unknown tool (expecting error)...")
        await mock_server.call_tool("unknown_tool", {})

    except McpError as e:
         print(f"Caught expected MCP Error: {e.error.code} - {e.error.message}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")

    # 调用其他方法...
    # prompts = await mock_server.list_prompts()
    # prompt_msgs = await mock_server.get_prompt("prompt_name", {"arg": "val"})
    # resource_contents = await mock_server.read_resource("resource://uri")
```

- `MockMcp` 会设置一个模拟的请求上下文，因此处理器内部调用 `request_ctx.get()` 可以工作，并能访问到捕获的 `lifespan_context`。
- 错误（如 `ValueError`）会被 `MockMcp` 捕获并包装成 `McpError`，类似于真实服务器的行为。

## MockFastMcp (用于 `fastmcp.server.server.FastMCP`)

`MockFastMcp` 模拟了更高级别的 `FastMCP` 服务器。

### 1. 创建和配置 `FastMCP` 实例

使用 `@app.tool()`, `@app.resource()`, `@app.prompt()` 装饰器创建你的 `FastMCP` 应用。

```python
# fastmcp_app.py
from fastmcp import FastMCP
from fastmcp.server.context import Context
from contextlib import asynccontextmanager
from typing import AsyncIterator

class AppState:
    counter: int = 0

@asynccontextmanager
async def app_lifespan(app: FastMCP[AppState]) -> AsyncIterator[AppState]:
    state = AppState()
    print(f"Lifespan Start: {app.name}")
    yield state
    print(f"Lifespan End: {app.name}")

app = FastMCP[AppState]("MyFastApp", lifespan=app_lifespan)

@app.tool()
def process_data(data: str, ctx: Context) -> str:
    """一个使用上下文的工具"""
    # 访问 lifespan 状态
    lifespan_state = ctx.lifespan # 类型为 AppState
    lifespan_state.counter += 1
    # 使用上下文进行日志记录
    ctx.info(f"Processing data: {data}. Counter: {lifespan_state.counter}")
    return f"Processed: {data} (Count: {lifespan_state.counter})"

@app.resource("resource://items/{item_id}")
def get_item(item_id: int) -> dict:
    return {"id": item_id, "description": f"Description for item {item_id}"}

# 可以挂载其他服务器
# base_app = FastMCP("Base")
# @base_app.tool()
# def base_util(): return "Base utility result"
# app.mount("base", base_app)
```

### 2. 将 `FastMCP` 映射到 `MockFastMcp`

使用 `map_fastmcp_to_mock` 异步函数。它会处理 lifespan、复制管理器（包括本地和导入的项），并递归地处理挂载的服务器（添加带前缀的项）。

```python
import asyncio
# 假设 mock_fastmcp.py 在 python 路径中
from mock_fastmcp import map_fastmcp_to_mock, MockFastMcp
# 从你的应用文件中导入 app 实例
from fastmcp_app import app, AppState

async def run_mock_fast():
    # 映射 FastMCP 到 MockFastMcp
    # 指定类型参数 [AppState]
    mock_app: MockFastMcp[AppState] = await map_fastmcp_to_mock(app)

    print(f"Mock App Name: {mock_app.name}")
    # 检查 lifespan 上下文
    print(f"Lifespan context counter: {mock_app._lifespan_context.counter}") # 初始为 0

    # ... 调用 mock app 的方法 ...

# if __name__ == "__main__":
#     asyncio.run(run_mock_fast())
```

### 3. 直接调用 Mock 方法

调用 `MockFastMcp` 实例上的方法。这些方法直接与内部的管理器（`ToolManager` 等）交互。

```python
# (在 run_mock_fast 函数内部)
    try:
        print("\nListing tools...")
        tools = await mock_app.get_tools() # 获取所有工具 (包括挂载的)
        print(f"Available tools: {list(tools.keys())}")

        print("\nCalling process_data tool...")
        result_content = await mock_app.call_tool("process_data", {"data": "abc"})
        print(f"Result: {result_content[0].text}") # call_tool 返回 content 列表

        # 再次调用，检查 lifespan 状态是否更新
        result_content = await mock_app.call_tool("process_data", {"data": "xyz"})
        print(f"Result: {result_content[0].text}")

        print("\nReading item resource...")
        resource_result = await mock_app.read_resource("resource://items/123")
        # read_resource 返回 ReadResourceContents 列表
        import json
        resource_data = json.loads(resource_result[0].content) # type: ignore
        print(f"Resource data: {resource_data}")

        # 调用挂载的工具 (如果上面取消了注释)
        # print("\nCalling mounted tool...")
        # mounted_result = await mock_app.call_tool("base_base_util", {})
        # print(f"Mounted result: {mounted_result[0].text}")

    except NotFoundError as e:
        print(f"Caught NotFoundError: {e}")
    except Exception as e:
        print(f"Caught unexpected error: {e}")
```

- **上下文注入**: 如果你的处理器函数（工具、资源、提示）通过类型提示请求 `Context`，`MockFastMcp` 会注入一个 `MockContext` 实例。你可以通过 `ctx.lifespan` 访问捕获的 lifespan 状态，并使用模拟的日志方法 (`ctx.info`, `ctx.debug` 等)。
- **挂载和导入**: `map_fastmcp_to_mock` 会自动处理 `app.mount()` 和 `app.import_server()`。挂载服务器的项会以正确的（可配置的）前缀添加到 mock 实例的管理器中。导入的项因为直接添加到了原服务器的管理器中，也会被复制过来。
- **错误处理**: `MockFastMcp` 会直接重新引发处理器或管理器中发生的异常（如 `NotFoundError`, `ValueError` 等）。

## 总结

`MockMcp` 和 `MockFastMcp` 提供了一种强大的方式来测试和本地运行你的 MCP/FastMCP 服务器逻辑，无需处理底层通信协议的复杂性。它们通过映射现有服务器实例来重用你的处理器和 lifespan 逻辑，确保了测试的一致性。
