import sys
import pathlib

import mockmcp

from mockmcp import map_fastmcp_to_mock
sys.path.append(str(pathlib.Path(__file__).parent.parent))
async def test_mock_fastmcp1():
    from tests.exist_mcp.fastmcp1 import create_echo_server
    fastmcp1=create_echo_server()
    mock_fastmcp1=await map_fastmcp_to_mock(fastmcp1)
    results=await mock_fastmcp1.call_tool("echo",{"text":"1"})
    print(results)


async def test_mock_fastmcp():
    from tests.exist_mcp.ppt_fastmcp import mcp as ppt_server
    mock_ppt_server=await map_fastmcp_to_mock(ppt_server)
    results=await mock_ppt_server.call_tool("create_presentation",{"title":"Test Presentation"})
    print(results)
    tools=await mock_ppt_server.get_tools()
    print(tools)
    resources=await mock_ppt_server.get_resources()
    print(resources)
    prompts=await mock_ppt_server.get_prompts()
    print(prompts)
    

import asyncio
if __name__ =="__main__":
    #asyncio.run(test_mock_fastmcp1())
    asyncio.run(test_mock_fastmcp())
