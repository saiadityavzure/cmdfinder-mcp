import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

os.environ.setdefault("USER_AGENT", "cmdfinder-mcp/1.0")

DOCS_URL = os.getenv("DOCS_URL")
if not DOCS_URL:
    raise ValueError("DOCS_URL is not set in .env file")

mcp = FastMCP(
    name="cmdfinder-mcp",
    instructions=(
        "cmdfinder-mcp helps you find the correct CLI command for a given "
        "description by searching live documentation pages. "
        "Use the find_command tool to search for commands by describing "
        "what you want to do. Use the get_docs_url tool to check which "
        "documentation source is currently configured."
    ),
)


@mcp.tool()
async def find_command(query: str) -> str:
    """
    Scrapes the configured documentation URL and returns the most relevant
    CLI command matching the natural language query.

    Args:
        query: Natural language description of what you want to do or find.
               Example: "show BGP neighbor status"
               Example: "display all VLANs configured on the switch"
               Example: "check interface errors and drops"
    """
    url = DOCS_URL

    # Step 1: Scrape the live docs page via WebBaseLoader
    try:
        loader = WebBaseLoader(
            web_paths=[url],
            bs_kwargs={"features": "html.parser"},
        )
        docs = []
        async for doc in loader.alazy_load():
            docs.append(doc)
    except Exception as e:
        return f"[cmdfinder-mcp] Failed to load documentation: {str(e)}"

    if not docs:
        return f"[cmdfinder-mcp] No content returned from: {url}"

    # Step 2: Split into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    # Step 3: Score and filter chunks by keyword relevance to the query
    query_words = [w.lower() for w in query.split() if len(w) > 2]

    scored_chunks = []
    for chunk in chunks:
        content_lower = chunk.page_content.lower()
        score = sum(1 for word in query_words if word in content_lower)
        if score > 0:
            scored_chunks.append((score, chunk.page_content))

    # Sort by highest relevance score
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Take top 5 most relevant chunks
    top_chunks = [content for _, content in scored_chunks[:5]]

    if not top_chunks:
        return (
            f"[cmdfinder-mcp] No relevant content found for: '{query}'\n"
            f"Try rephrasing with specific keywords from the documentation."
        )

    context = "\n\n---\n\n".join(top_chunks)

    # Step 4: Return structured context for the LLM to reason over
    return f"""
QUERY: {query}
SOURCE: {url}

RELEVANT DOCUMENTATION SECTIONS:

{context}

---
Based on the above documentation, identify the most relevant command
matching the query. Return:
1. The exact command syntax
2. What it displays or does
3. Any important optional parameters
"""


@mcp.tool()
async def get_docs_url() -> str:
    """
    Returns the documentation URL currently configured in the .env file.
    Useful to confirm which CLI reference source cmdfinder-mcp is using.
    """
    return f"[cmdfinder-mcp] Active docs URL: {DOCS_URL}"


if __name__ == "__main__":
    print("=" * 40)
    print("  cmdfinder-mcp")
    print("=" * 40)
    print(f"  Docs URL : {DOCS_URL}")
    print(f"  Tools    : find_command, get_docs_url")
    print(f"  Transport: SSE")
    print(f"  SSE URL  : http://0.0.0.0:9008/sse")
    print("=" * 40)
    mcp.run(transport="sse", host="0.0.0.0", port=9008)
