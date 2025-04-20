from fastmcp import FastMCP

def create_echo_server():
    # Create server
    mcp = FastMCP("Echo Server")


    @mcp.tool()
    def echo(text: str) -> str:
        """Echo the input text"""
        return text

    return mcp