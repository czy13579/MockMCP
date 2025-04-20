import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import mcp.types as types
from mcp.server.lowlevel.server import Server
from mcp.shared.exceptions import McpError
import asyncio

def create_calculator_server():
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
    
    return calculator_server
