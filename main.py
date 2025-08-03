import asyncio
import os
from dotenv import load_dotenv

from mcp_server.client import agents  # Adjust this import path if needed

async def main():
    load_dotenv()
    llm_model = os.getenv("OLLAMA_LLM")  # Replace with your actual model name
    llm_provider = "ollama"  # or "aws"
    question = "What are the latest advancements in quantum computing?"

    try:
        response = await agents(llm_model, llm_provider, question)
        print(f"\nFinal Agent Response:\n{response}")
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == '__main__':
    asyncio.run(main())
