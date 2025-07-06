import asyncio
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_core import CancellationToken
import os 


os.environ["OPENAI_API_KEY"] = ""

async def main() -> None:
    # Setup server params for local filesystem access
    # desktop = str(Path.home() / "Desktop")
    desktop = "/home/sarveshharikant/EXIMIETAS/git_AWS/aws/"
    server_params = StdioServerParams(
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", desktop]
    )

    # Get all available tools from the server
    tools = await mcp_server_tools(server_params)

    # Create an agent that can use all the tools
    agent = AssistantAgent(
        name="file_manager",
        model_client=OpenAIChatCompletionClient(model="gpt-4"),
        tools=tools,  # type: ignore
    )

    # The agent can now use any of the filesystem tools
    await agent.run(task="Create a file called test.txt by adding virat kohliii name", cancellation_token=CancellationToken())


if __name__ == "__main__":
    asyncio.run(main())
