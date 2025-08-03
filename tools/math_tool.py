# ## local math calculations


import math
from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    name="math", # only used for SSE transport (set this to any port)
    # stateless_http=True,
)


@mcp.tool()
def add(a: float, b: float) -> float:
    """calculate and returns the sum of two numbers.
    """
    print(f"Server received add request: {a}, {b}")
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """calculate and returns the subtraction of two numbers."""
    print(f"Server received subtract request: {a}, {b}")
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """calculate and returns multiply of two numbers."""
    print(f"Server received multiply request: {a}, {b}")
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """calculate and returns the dividing of two numbers."""
    print(f"Server received divide request: {a}, {b}")
    if b != 0:
        return a / b
    return math.nan


@mcp.tool()
def sine(a: float) -> float:
    """calculate and returns the sine of a number."""
    print(f"Server received sine request: {a}")
    return math.sin(a)


if __name__ == "__main__":
    print("Running server with stdio transport for math calculations...")
    mcp.run(transport="stdio")
