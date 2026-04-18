"""
================================================================================
                         UNIFIED STREAMLIT FRONTEND
================================================================================

This is the ONE frontend file that replaces all 7 separate frontend files:

  ┌─────────────────────────────────┬─────────────────────────────────────────┐
  │  Old File                       │  Feature it had                         │
  ├─────────────────────────────────┼─────────────────────────────────────────┤
  │  streamlit_frontend.py          │  Basic chat (no streaming)              │
  │  streamlit_frontend_streaming.py│  Token-by-token streaming               │
  │  streamlit_frontend_threading.py│  Tool filtering (show only AI tokens)   │
  │  streamlit_frontend_database.py │  SQLite persistence + sidebar threads   │
  │  streamlit_frontend_tool.py     │  Tool status indicators (🔧 spinner)    │
  │  streamlit_frontend_mcp.py      │  MCP tools + async streaming            │
  │  streamlit_rag_frontend.py      │  PDF upload + RAG Q&A                   │
  └─────────────────────────────────┴─────────────────────────────────────────┘

  ALL of these features are now combined here in a single, clean file.

HOW TO RUN:
    streamlit run app.py

FILE STRUCTURE (6 sections):
    Section 1 — Imports & Page Config
    Section 2 — Utility Functions
    Section 3 — Session State Initialization
    Section 4 — Sidebar (mode selector, new chat, PDF upload, thread list)
    Section 5 — Main Chat Area (history, input, streaming response)
    Section 6 — Thread Switching Logic

================================================================================
"""

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — IMPORTS & PAGE CONFIG                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import uuid  # For generating unique thread IDs

import streamlit as st  # The UI framework

# --- Import from our unified backend ---
# chatbot:                  The compiled LangGraph graph (call .stream() on it)
# retrieve_all_threads:     Gets all thread IDs from the SQLite database
# ingest_pdf:               Processes a PDF and creates a searchable vector index
# thread_document_metadata: Gets PDF info (filename, pages, chunks) for a thread
from unified_backend import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf,
    thread_document_metadata,
)

# --- LangChain message types ---
# We need these to identify what kind of message is streaming through:
#   HumanMessage → user typed this
#   AIMessage    → LLM generated this (what we show to the user)
#   ToolMessage  → result of a tool call (we show a status indicator for these)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# --- Page Configuration ---
# This MUST be the first Streamlit command in the file
st.set_page_config(
    page_title="LangGraph Chatbot",      # Browser tab title
    page_icon="🤖",                       # Browser tab icon
    layout="centered",                    # Center the chat in the page
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — UTILITY FUNCTIONS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# These are helper functions used by the sidebar and main chat area.
# They handle thread management and conversation loading.


def generate_thread_id():
    """
    Generate a unique thread ID using UUID4.

    Every conversation gets its own thread_id. This ID is used:
      - As the key in LangGraph's checkpointer (SQLite) to save/load state
      - In the sidebar to list and switch between conversations
      - To associate uploaded PDFs with specific conversations

    Returns:
        A UUID4 object (will be converted to string when needed).
    """
    return uuid.uuid4()


def reset_chat():
    """
    Start a brand new conversation.

    Called when the user clicks "New Chat" in the sidebar. It:
      1. Generates a fresh thread_id
      2. Registers it in the thread list
      3. Clears the message history so the chat area is empty

    Note: The old conversation is NOT deleted — it's still saved in SQLite
    and can be accessed from the sidebar.
    """
    new_id = generate_thread_id()
    st.session_state["thread_id"] = new_id
    add_thread(new_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    """
    Add a thread_id to the sidebar list if it's not already there.

    This prevents duplicates when the same thread is referenced multiple times
    (e.g., on page reloads).

    Args:
        thread_id: The thread ID to add.
    """
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    """
    Load all messages for a given thread from the LangGraph checkpointer.

    How it works:
      - chatbot.get_state() reads the saved state from SQLite for this thread
      - The state contains a "messages" key with all HumanMessage/AIMessage objects
      - If the thread has no messages yet, returns an empty list

    Args:
        thread_id: The thread to load.

    Returns:
        A list of LangChain message objects (HumanMessage, AIMessage, etc.)
    """
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — SESSION STATE INITIALIZATION                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# Streamlit reruns this entire script on every user interaction (click, input, etc.)
# To keep data between reruns, we store it in st.session_state (a dict-like object).
#
# The "if key not in st.session_state" pattern ensures we only initialize once
# (on the very first page load), not on every rerun.

# --- message_history ---
# A list of dicts like: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
# This is the SOURCE OF TRUTH for what's displayed in the chat area.
# We maintain our own list (instead of reading from LangGraph every time) for speed.
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# --- thread_id ---
# The currently active conversation thread.
# Every message sent goes to this thread in the backend.
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# --- chat_threads ---
# List of all thread IDs (for the sidebar).
# On first load, we query SQLite to get any previously saved threads.
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# --- ingested_docs ---
# Tracks which PDFs have been uploaded for which threads.
# Structure: { "thread-id-string": { "filename.pdf": {metadata} } }
# This prevents re-processing the same PDF if the user uploads it again.
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# --- selected_thread ---
# Temporary flag set when the user clicks a thread in the sidebar.
# We process this AFTER rendering the chat to avoid display glitches.
if "selected_thread" not in st.session_state:
    st.session_state["selected_thread"] = None

# Make sure the current thread is in the sidebar list
add_thread(st.session_state["thread_id"])

# Convenience variables (shorter names for frequently accessed values)
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — SIDEBAR                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# The sidebar contains:
#   1. App title and current thread ID
#   2. "New Chat" button
#   3. Mode selector (Chat / Tools / RAG)
#   4. PDF upload widget (only visible in RAG mode)
#   5. List of past conversation threads

# --- 4a. Title & Thread Info ---
st.sidebar.title("🤖 LangGraph Chatbot")
st.sidebar.caption(f"Thread: `{thread_key[:8]}…`")  # Show first 8 chars of thread ID

# --- 4b. New Chat Button ---
# use_container_width=True makes the button span the full sidebar width
if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()  # Force a page reload to clear the chat area

st.sidebar.divider()  # Visual separator

# --- 4c. Mode Selector ---
# This lets the user switch the UI context. Under the hood, the LLM ALWAYS has
# access to all tools. The mode just changes:
#   - The chat input placeholder text
#   - Whether the PDF upload widget is shown
#   - A subtle visual indicator of what mode you're in
#
# We use a radio button with horizontal layout to save sidebar space.
mode = st.sidebar.radio(
    "Mode",
    options=["💬 Chat", "🔧 Tools", "📄 RAG"],
    horizontal=True,
    help="All tools are always available. This just changes the UI focus.",
)

# --- 4d. PDF Upload (RAG mode only) ---
# Only show the PDF upload widget when the user is in RAG mode.
# This keeps the UI clean in other modes.
if mode == "📄 RAG":
    # Show current document status
    if thread_docs:
        # A PDF has been uploaded for this thread — show its info
        latest_doc = list(thread_docs.values())[-1]
        st.sidebar.success(
            f"📎 **{latest_doc.get('filename')}**\n\n"
            f"{latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages"
        )
    else:
        # No PDF yet — prompt the user
        st.sidebar.info("No PDF indexed yet. Upload one below.")

    # PDF file uploader widget
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF for this chat",
        type=["pdf"],
        help="The PDF will be split into chunks and indexed for search.",
    )

    # Process the uploaded PDF (only if it hasn't been processed already)
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            # Already processed — don't re-process
            st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
        else:
            # Show a progress indicator while processing
            with st.sidebar.status("📑 Indexing PDF…", expanded=True) as status_box:
                # Call the backend to ingest the PDF
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),     # Raw bytes of the PDF
                    thread_id=thread_key,        # Associate with current thread
                    filename=uploaded_pdf.name,   # Original filename
                )
                # Save the result so we don't re-process
                thread_docs[uploaded_pdf.name] = summary
                # Update the status indicator
                status_box.update(
                    label="✅ PDF indexed successfully",
                    state="complete",
                    expanded=False,
                )

st.sidebar.divider()  # Visual separator

# --- 4e. Past Conversations List ---
st.sidebar.subheader("📂 Past Conversations")

# Get the thread list in reverse order (newest first)
threads = st.session_state["chat_threads"][::-1]

if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        # Each thread gets a button. When clicked, we set selected_thread
        # and process the switch AFTER rendering (see Section 6).
        # The key parameter ensures each button is unique (Streamlit requirement).
        if st.sidebar.button(
            f"💬 {str(tid)[:8]}…",
            key=f"thread-btn-{tid}",
            use_container_width=True,
        ):
            st.session_state["selected_thread"] = tid


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MAIN CHAT AREA                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# This section handles:
#   a) Displaying the full message history
#   b) The chat input box
#   c) Sending the user's message to the backend
#   d) Streaming the assistant's response token-by-token
#   e) Showing tool status indicators when the LLM uses a tool

# --- 5a. Page Title ---
# Change the title based on the selected mode
mode_titles = {
    "💬 Chat": "💬 Chat",
    "🔧 Tools": "🔧 Tools & Search",
    "📄 RAG": "📄 Document Q&A",
}
st.title(mode_titles.get(mode, "💬 Chat"))

# --- 5b. Render Message History ---
# Loop through all saved messages and display them as chat bubbles.
# st.chat_message() creates a chat bubble with the appropriate avatar.
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        # Use st.markdown() instead of st.text() for rich formatting
        # (the LLM often returns markdown-formatted text)
        st.markdown(message["content"])

# --- 5c. Chat Input ---
# The placeholder text changes based on the selected mode
mode_placeholders = {
    "💬 Chat": "Type your message…",
    "🔧 Tools": "Ask me to search, calculate, or check stock prices…",
    "📄 RAG": "Ask about your uploaded document…",
}
user_input = st.chat_input(mode_placeholders.get(mode, "Type here…"))

# --- 5d. Process User Input ---
if user_input:

    # Step 1: Add the user's message to our local history
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Step 2: Display the user's message immediately (before waiting for the LLM)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Step 3: Build the LangGraph config
    # The config tells LangGraph which thread to use for persistence
    CONFIG = {
        "configurable": {"thread_id": thread_key},  # Which thread in SQLite
        "metadata": {"thread_id": thread_key},       # Extra metadata for logging
        "run_name": "chat_turn",                      # Name for tracing/debugging
    }

    # Step 4: Stream the assistant's response
    # We use a generator function + st.write_stream() for token-by-token display.
    with st.chat_message("assistant"):

        # status_holder is a mutable dict so the generator can create/update
        # a Streamlit status widget from inside the generator.
        # (We can't use a simple variable because generators have their own scope.)
        status_holder = {"box": None}

        def ai_only_stream():
            """
            Generator that yields ONLY the AI's text tokens.

            How streaming works in LangGraph:
              - chatbot.stream() yields (message_chunk, metadata) pairs
              - message_chunk can be:
                  • AIMessage    → text the LLM is generating (we YIELD these)
                  • ToolMessage  → result of a tool call (we show a status indicator)
                  • HumanMessage → the user's input echoed back (we SKIP these)

            The key insight: we only yield AIMessage content to st.write_stream().
            Tool messages are handled by creating/updating a st.status() widget.
            """
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",  # Stream individual message chunks
            ):

                # --- Handle Tool Messages ---
                # When the LLM calls a tool, the tool's result comes back as a ToolMessage.
                # We show a visual indicator so the user knows a tool is running.
                if isinstance(message_chunk, ToolMessage):
                    # Get the tool name (e.g., "calculator", "duckduckgo_search")
                    tool_name = getattr(message_chunk, "name", "tool")

                    if status_holder["box"] is None:
                        # First tool call → create a new status widget
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…",
                            expanded=True,
                        )
                    else:
                        # Subsequent tool call → update the existing status widget
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                # --- Handle AI Messages (the actual response) ---
                # Only yield AI text tokens — this is what appears in the chat bubble.
                # We skip empty content (which happens for tool-call-only messages).
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        # st.write_stream() takes a generator and displays tokens as they arrive.
        # It returns the complete concatenated text when done.
        ai_message = st.write_stream(ai_only_stream())

        # Finalize the tool status indicator (if any tools were used)
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )

    # Step 5: Save the complete assistant message to our local history
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Step 6 (RAG mode only): Show document metadata below the response
    if mode == "📄 RAG":
        doc_meta = thread_document_metadata(thread_key)
        if doc_meta:
            st.caption(
                f"📎 Source: {doc_meta.get('filename')} "
                f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — THREAD SWITCHING LOGIC                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# When the user clicks a thread in the sidebar, we need to:
#   1. Update the current thread_id
#   2. Load that thread's messages from SQLite
#   3. Convert them to our local format (role + content dicts)
#   4. Reload the page to show the loaded conversation
#
# WHY is this at the bottom?
#   Because Streamlit runs top-to-bottom. If we switched threads in the sidebar
#   section (Section 4), the chat area would render with the OLD messages first,
#   then we'd switch — causing a visual flash. By doing it here and calling
#   st.rerun(), the page reloads cleanly with the correct messages.

if st.session_state["selected_thread"] is not None:
    selected = st.session_state["selected_thread"]

    # Update the active thread
    st.session_state["thread_id"] = selected

    # Load messages from the backend (SQLite checkpointer)
    messages = load_conversation(selected)

    # Convert LangChain message objects to our simple dict format
    # We only care about two roles: "user" and "assistant"
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    # Replace the message history with the loaded conversation
    st.session_state["message_history"] = temp_messages

    # Initialize the ingested_docs entry for this thread (empty if new)
    st.session_state["ingested_docs"].setdefault(str(selected), {})

    # Clear the selection flag so we don't re-trigger on the next rerun
    st.session_state["selected_thread"] = None

    # Reload the page to display the loaded conversation
    st.rerun()
