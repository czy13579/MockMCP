import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))
from mockmcp.mock_mcp import map_server_to_mock

from tests.exist_mcp.mcp1 import create_calculator_server
from tests.exist_mcp.mcp2 import create_e2b_code_mcp_server
from tests.exist_mcp.mcp3 import create_time_mcp_server
import asyncio
async def test_mock_mcp1():
    calculator_server = create_calculator_server()
    mock_mcp = await map_server_to_mock(calculator_server)
    tools = await mock_mcp.list_tools()
    print(tools)

async def test_mock_mcp2():
    code_mcp_server = create_e2b_code_mcp_server()
    mock_mcp = await map_server_to_mock(code_mcp_server)
    tools = await mock_mcp.list_tools()
    print(tools)
    results=await mock_mcp.call_tool("run_code",{"code":"print('Hello, World!')"})
    print(results)

async def test_mock_mcp3():
    time_mcp_server = create_time_mcp_server()
    mock_mcp = await map_server_to_mock(time_mcp_server)
    tools = await mock_mcp.list_tools()
    print(tools)

if __name__ == "__main__":
    #asyncio.run(test_mock_mcp1())
    #asyncio.run(test_mock_mcp2())
    asyncio.run(test_mock_mcp3())

