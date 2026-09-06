"""
Phase 5: MCP server wrapping retrieve_context().
Compatible with mcp 2.x (using MCPServer).
"""

import json
from mcp.server.mcpserver import MCPServer
from phase3_retrieve_context import retrieve_context

# Initialize MCPServer instance
mcp = MCPServer("bank-filings-qa-server")


@mcp.tool()
def retrieve_bank_filings(query: str, k: int = 5) -> str:
  """Search real SEC 10-K filings from JPMorgan, Bank of America, and Wells Fargo

  for text relevant to a financial question. Returns the most relevant excerpts
  with their source (ticker and filing date).
  """
  results = retrieve_context(query, k=k)

  if not results:
    return "No relevant filings found for this query."

  return json.dumps(results, indent=2)


if __name__ == "__main__":
  mcp.run()