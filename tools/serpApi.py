import asyncio
import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from langchain_core.messages import ToolMessage

# Initialize MCP tool server
mcp = FastMCP(name="serpsearch", host="localhost", port=8001)

class SerpApiSearch:
    def __init__(self):
        load_dotenv()
        self._API_KEY = os.getenv("SERP_API")
        self._BASE_URL = os.getenv("SERP_URL")

    @mcp.tool()
    async def search_serpapi(self, query: str) -> ToolMessage:
        """
        Use this tool to search the internet for up-to-date information.
        It fetches real-time data using SerpAPI and returns relevant snippets.
        """

        if not self._BASE_URL or not self._API_KEY:
            return ToolMessage(
                content="Missing SERP_API or SERP_URL in environment variables.",
                tool_call_id="serp_tool"
            )

        params = {"q": query, "api_key": self._API_KEY}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self._BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                snippets = self.extract_snippets(data)
                return ToolMessage(
                    content="\n".join(snippets) if snippets else "No relevant snippets found.",
                    tool_call_id="serp_tool"
                )
        except httpx.ReadTimeout:
            return ToolMessage(content="Request to SerpAPI timed out.", tool_call_id="serp_tool")
        except httpx.HTTPStatusError as e:
            return ToolMessage(
                content=f"HTTP error: {e.response.status_code} - {e.response.text}",
                tool_call_id="serp_tool"
            )
        except Exception as e:
            return ToolMessage(content=f"Unexpected error: {str(e)}", tool_call_id="serp_tool")

    @staticmethod
    def extract_snippets(data: dict):
        if "error" in data:
            return [data["error"]]
        return [
            result.get("snippet", "")
            for result in data.get("organic_results", [])
            if result.get("snippet")
        ]


# Optional: test the tool directly
async def main():
    serp = SerpApiSearch()
    result = await serp.search_serpapi("Who is the current USA president?")
    print(result.content)

if __name__ == "__main__":
    #asyncio.run(main())
    serp = SerpApiSearch()
    mcp.run(transport="streamable-http")
