from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from models.ollama_model import OllamaLLM
from models.bedrock_model import BedrockLLM
import traceback


async def agents(llm_model, llm_provider, question):
    try:
        print(
            f"Setting up MCP Client with model: {llm_model} and provider: {llm_provider}"
        )
        if llm_provider == "aws":
            model = BedrockLLM().get_llm()
        elif llm_provider == "ollama":
            model = OllamaLLM().get_llm()

    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM: {e}")


    mcp_client = MultiServerMCPClient(
        {
            "serpsearch": {
                "url": "http://localhost:8001/mcp/",
                "transport": "streamable_http",
            },
            "promptgen": {
                "url": "http://localhost:8002/mcp/",
                "transport": "streamable_http",
            },
            # "math": {
            #     "command": "python",
            #     "args": ["tools/math_tool.py"],
            #     "transport": "stdio",
            # },
        }
    )

    print("Connecting to MCP tools and prompts")

    try:
        # Load tools
        tools = await mcp_client.get_tools()

        # Load prompts individually
        security_prompt_msg = await mcp_client.get_prompt(
            "promptgen", "security_prompt"
        )
        system_prompt_msg = await mcp_client.get_prompt("promptgen", "system_prompt")

        # If it's a list, take the first item
        if isinstance(security_prompt_msg, list):
            security_prompt_msg = security_prompt_msg[0]
        if isinstance(system_prompt_msg, list):
            system_prompt_msg = system_prompt_msg[0]

        # Extract content
        security_prompt = security_prompt_msg.content
        system_prompt = system_prompt_msg.content

    except* Exception as eg:
        print("ExceptionGroup caught during tool/prompt loading:")
        traceback.print_exception(eg)
        raise RuntimeError(f"Failed to load tools or prompts: {eg}")

    print(f"Loaded Tools: {[tool.name for tool in tools]}")
    #Binding LLM with tools

    # using init_chat_model and .bind_tools as a more general solution
    agent = init_chat_model(model=llm_model, model_provider=llm_provider, temperature=0).bind_tools(tools)

    response = await agent.ainvoke(
        {
            "messages": [
                {"role": "system", "content": security_prompt},
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
        }
    )

    # print(f"Response: {response}")
    print(f"Agent Response: {response['messages'][-1].content}")
    return response["messages"][-1].content

