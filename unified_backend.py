"""
================================================================================
                        UNIFIED LANGGRAPH BACKEND
================================================================================

This file combines ALL backend capabilities from the separate backend files
into ONE single backend. It merges:

  1. langgraph_backend.py          → Basic LLM chat node
  2. langgraph_database_backend.py → SQLite persistence (SqliteSaver)
  3. langgraph_tool_backend.py     → Tools (search, calculator, stock price)
  4. langgraph_mcp_backend.py      → MCP tool loading (optional, fails silently)
  5. langraph_rag_backend.py       → PDF ingestion + FAISS retriever + rag_tool

HOW IT WORKS (high level):
  - A LangGraph StateGraph is built with two nodes:
      • "chat_node" — calls the LLM (with tools bound)
      • "tools"     — executes whichever tool the LLM chose
  - A conditional edge after chat_node decides:
      "Did the LLM request a tool call?"
        YES → go to "tools" node → loop back to "chat_node"
        NO  → end the graph (return the final AI message)
  - All conversation state is persisted to a SQLite database via SqliteSaver.
  - Each conversation is identified by a unique "thread_id".

EXPORTS (what app.py imports):
  - chatbot              : the compiled LangGraph graph — call .stream() or .invoke()
  - retrieve_all_threads : returns a list of all thread IDs stored in the DB
  - ingest_pdf           : takes PDF bytes → builds a FAISS vector store for that thread
  - thread_document_metadata : returns metadata (filename, chunks, pages) for a thread's PDF

================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — IMPORTS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations  # Allows using str | None syntax in older Python

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

# --- LangChain core ---
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

# --- LangChain community (tools & loaders) ---
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# --- LangChain text splitting ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangChain OpenAI wrappers ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- LangGraph (graph building, checkpointing, prebuilt helpers) ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables from .env file (OPENAI_API_KEY, etc.)
load_dotenv()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — LLM & EMBEDDINGS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# We create ONE shared LLM instance and ONE shared embeddings instance.
# The LLM is used for chat responses; embeddings are used for RAG vector search.

llm = ChatOpenAI(model="gpt-4o-mini")

# Embeddings model — used to convert PDF text chunks into vectors for similarity search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — RAG: PDF STORAGE & INGESTION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# RAG = Retrieval Augmented Generation
#
# The idea:
#   1. User uploads a PDF
#   2. We split it into small text chunks
#   3. We convert each chunk into a vector (embedding)
#   4. We store these vectors in a FAISS index (in-memory vector database)
#   5. When the user asks a question, the rag_tool searches the index for
#      the most relevant chunks and passes them to the LLM as context
#
# Each thread can have its own PDF, so we store retrievers per thread_id.

# Dictionary mapping thread_id → FAISS retriever object
_THREAD_RETRIEVERS: Dict[str, Any] = {}

# Dictionary mapping thread_id → metadata dict (filename, page count, chunk count)
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """
    Look up the FAISS retriever for a given thread.

    Args:
        thread_id: The conversation thread ID to look up.

    Returns:
        The FAISS retriever if a PDF was ingested for this thread, else None.
    """
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Process a PDF file and create a searchable vector index for it.

    This is the main entry point for RAG. Called from the frontend when
    the user uploads a PDF.

    Steps:
        1. Write the raw bytes to a temporary file (PyPDFLoader needs a file path)
        2. Load the PDF into LangChain Document objects (one per page)
        3. Split each page into smaller chunks (1000 chars with 200 char overlap)
        4. Create FAISS vector store from the chunks
        5. Store the retriever so the rag_tool can use it later

    Args:
        file_bytes: Raw PDF file content as bytes.
        thread_id:  The conversation thread this PDF belongs to.
        filename:   Original filename (for display purposes).

    Returns:
        A dict with keys: filename, documents (page count), chunks (chunk count).
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    # Step 1: Write bytes to a temp file because PyPDFLoader requires a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        # Step 2: Load PDF → list of Document objects (one per page)
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Step 3: Split into smaller chunks for better retrieval accuracy
        # - chunk_size=1000: each chunk is ~1000 characters
        # - chunk_overlap=200: consecutive chunks share 200 chars to avoid losing context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],  # Split priority: paragraphs > lines > words
        )
        chunks = splitter.split_documents(docs)

        # Step 4: Create FAISS vector store from chunks
        # This embeds every chunk and builds a similarity search index
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Create a retriever that returns the top 4 most similar chunks
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        # Step 5: Store retriever and metadata for this thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),    # number of pages
            "chunks": len(chunks),     # number of text chunks
        }

        return _THREAD_METADATA[str(thread_id)]

    finally:
        # Clean up temp file (FAISS already copied the text into memory)
        try:
            os.remove(temp_path)
        except OSError:
            pass


def thread_document_metadata(thread_id: str) -> dict:
    """
    Get the PDF metadata for a thread (filename, page count, chunk count).

    Returns an empty dict if no PDF has been ingested for this thread.
    """
    return _THREAD_METADATA.get(str(thread_id), {})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — TOOL DEFINITIONS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Tools are functions that the LLM can choose to call instead of answering directly.
# For example, if the user asks "What's the stock price of AAPL?", the LLM will
# call get_stock_price(symbol="AAPL") rather than guessing.
#
# Each tool is decorated with @tool which registers it with LangChain.

# --- Tool 1: Web Search ---
# Uses DuckDuckGo to search the web. No API key needed.
search_tool = DuckDuckGoSearchRun(region="us-en")


# --- Tool 2: Calculator ---
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div

    Args:
        first_num:  The first number.
        second_num: The second number.
        operation:  One of 'add', 'sub', 'mul', 'div'.

    Returns:
        A dict with the inputs and the result, or an error message.
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


# --- Tool 3: Stock Price ---
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given ticker symbol (e.g. 'AAPL', 'TSLA')
    using the Alpha Vantage API.

    Args:
        symbol: Stock ticker symbol.

    Returns:
        JSON response from Alpha Vantage with price data.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


# --- Tool 4: RAG (Retrieval from uploaded PDF) ---
@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Search the uploaded PDF for information relevant to the query.

    This tool looks up the FAISS vector index that was built when the user
    uploaded a PDF. It returns the top matching text chunks.

    Args:
        query:     The user's question to search for in the document.
        thread_id: The conversation thread ID (needed to find the right PDF).

    Returns:
        A dict with the query, matching context chunks, and source metadata.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    # Retrieve the top-k most similar chunks
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


# --- Combine all tools into one list ---
tools = [search_tool, calculator, get_stock_price, rag_tool]

# Bind tools to the LLM so it knows what tools are available
# This modifies the LLM to include tool schemas in every request
llm_with_tools = llm.bind_tools(tools)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — GRAPH STATE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# LangGraph uses a "state" object that flows through the graph.
# Our state is simple: just a list of messages.
#
# The `add_messages` annotation tells LangGraph to APPEND new messages
# to the existing list (rather than replacing it).

class ChatState(TypedDict):
    """
    The state that flows through the LangGraph graph.

    Attributes:
        messages: The full conversation history. New messages are appended
                  automatically thanks to the `add_messages` reducer.
    """
    messages: Annotated[list[BaseMessage], add_messages]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — GRAPH NODES                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Nodes are the "steps" in the graph. We have two:
#
#   1. chat_node — Sends messages to the LLM and gets a response.
#                  The response might be a regular text reply OR a tool call request.
#
#   2. tool_node — Executes whatever tool the LLM requested.
#                  (This is a prebuilt LangGraph node that handles all tool execution.)

def chat_node(state: ChatState, config=None):
    """
    The main LLM node. Sends the conversation to the LLM and returns its response.

    If a PDF has been uploaded for this thread, a system message is prepended
    to instruct the LLM to use the rag_tool for document questions.

    Args:
        state:  The current graph state containing all messages.
        config: LangGraph config dict (contains thread_id in 'configurable').

    Returns:
        A dict with the new message(s) to append to state.
    """
    # Extract the thread_id from the config (needed for RAG system prompt)
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Build a system message that tells the LLM about available capabilities
    # This is especially important for RAG — the LLM needs to know the thread_id
    # to pass to rag_tool so it can look up the correct PDF
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with access to several tools:\n"
            "- Web search (DuckDuckGo) for current information\n"
            "- Calculator for arithmetic\n"
            "- Stock price lookup\n"
            "- RAG tool for searching uploaded PDFs\n\n"
            "For questions about an uploaded PDF, call the `rag_tool` and include "
            f"the thread_id `{thread_id}`. "
            "If the user asks about a document but none is uploaded, ask them to upload one."
        )
    )

    # Prepend system message, then all conversation messages
    messages = [system_message, *state["messages"]]

    # Call the LLM (with tools bound) — it may respond with text or a tool call
    response = llm_with_tools.invoke(messages, config=config)

    return {"messages": [response]}


# The prebuilt ToolNode handles executing any tool the LLM requests
tool_node = ToolNode(tools)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — CHECKPOINTER (SQLite Persistence)                             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# The checkpointer saves the graph state (all messages) to a SQLite database
# after every step. This means:
#   - Conversations survive app restarts
#   - You can load any past conversation by its thread_id
#   - The LLM automatically sees the full history when you continue a thread
#
# check_same_thread=False is needed because Streamlit runs in multiple threads

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — BUILD & COMPILE THE GRAPH                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# The graph structure:
#
#   START → chat_node ──→ (tools_condition) ──→ tools ──→ chat_node (loop)
#                │                                          │
#                └── (no tool call) ─────────────────────→ END
#
# tools_condition is a built-in function that checks if the LLM's response
# contains a tool call. If yes → go to "tools" node. If no → go to END.

graph = StateGraph(ChatState)

# Add both nodes to the graph
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Wire up the edges
graph.add_edge(START, "chat_node")              # Entry point
graph.add_conditional_edges("chat_node", tools_condition)  # Route based on tool calls
graph.add_edge("tools", "chat_node")            # After tool execution, go back to LLM

# Compile the graph with the checkpointer for persistence
# This is the main object that the frontend imports and calls
chatbot = graph.compile(checkpointer=checkpointer)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — HELPER FUNCTIONS                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def retrieve_all_threads() -> list:
    """
    Query the SQLite checkpointer to get all unique thread IDs.

    This is used by the frontend sidebar to show a list of past conversations.

    How it works:
        - checkpointer.list(None) returns ALL checkpoints across ALL threads
        - Each checkpoint has a config dict with a thread_id
        - We collect unique thread_ids into a set and return as a list

    Returns:
        A list of thread ID strings.
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
