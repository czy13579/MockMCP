## 作用
快速将使用mcp、fastmcp实现的mcp_server转化为agent可调用的工具，无需docker等配置。


## 使用方法
1.安装mockmcp
```python
pip install mockmcp
```

2.将mcp_server转化为一个可以直接调用的类
```python
## 转化fastmcp

# server代码
from fastmcp import FastMCP
mcp = FastMCP("Echo Server")
@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text"""
    return text

# 进行转化
from mockmcp import map_fastmcp_to_mock
mock_mcp=await map_fastmcp_to_mock(mcp)
results=await mock_mcp.call_tool("echo",{"text":"1"})
print(results)
```

```python
## 转化mcp.Server

# server代码
calculator_server = Server("my_tool_server")

@calculator_server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="calculator",
            description="Performs simple calculations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression"},
                },
                "required": ["expression"],
            },
        )
    ]

@calculator_server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name == "calculator":
        if not arguments or "expression" not in arguments:
            raise ValueError("Missing 'expression' argument for calculator")
        try:
            # WARNING: eval is unsafe in real applications! This is just an example.
            result = eval(arguments["expression"], {"__builtins__": {}}, {})
            return [types.TextContent(type="text", text=str(result))]
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    else:
        raise ValueError(f"Unknown tool: {name}")

# 进行转化
from mockmcp import map_server_to_mock
mock_mcp=await map_server_to_mock(mcp)
tools=await mock_mcp.list_tools()
print(tools)
```

