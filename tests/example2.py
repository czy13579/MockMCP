# server代码
from mcp.server.lowlevel.server import Server
import mcp.types as types
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
async def main():   
    mock_mcp=await map_server_to_mock(calculator_server)
    tools=await mock_mcp.list_tools()
    print(tools)
    results=await mock_mcp.call_tool("calculator",{"expression":"1+1"})
    print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())