import asyncio
import json
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from tools.mcp_service.arxiv_service import ArxivMCPClient

async def main():
    mcp_client = ArxivMCPClient()
    result = await mcp_client.call_tool(
        "search_papers",
        {
        "query": "AbsGS",
        # "max_results": 10,
        # "categories": ["cs.LG", "cs.CL"],
        }
    )
    results = json.loads(result.content[0].text)
    resultsss = await mcp_client.call_tool(
        "download_paper",
        {
            "paper_id": results['papers'][0]['id']
        }
    )

    
    mcp_tool_spec = McpToolSpec(client=mcp_client)
    
    # tools = await mcp_tool_spec.to_tool_list_async()
    # print(tools)

if __name__ == "__main__":
    asyncio.run(main())