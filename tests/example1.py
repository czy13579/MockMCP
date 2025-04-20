# server代码
from fastmcp import FastMCP
mcp = FastMCP("Echo Server")
@mcp.tool()
def echo(text: str) -> str:
    """Echo the input text"""
    return text

# 进行转化
async def main():
    from mockmcp import map_fastmcp_to_mock
    mock_mcp=await map_fastmcp_to_mock(mcp)
    results=await mock_mcp.call_tool("echo",{"text":"1"})
    print(results)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())