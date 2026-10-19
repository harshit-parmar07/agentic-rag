import os
from typing import List, Literal, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from duckduckgo_search import DDGS

from config import GROQ_API_KEY, GROQ_MODEL
from vectorstore import get_retriever



@tool
def rag_search_tool(query: str, namespace: str = "default") -> str:
    """Top-K chunks from session-scoped KB with source and page citations (empty string if none)"""
    try:
        retriever_instance = get_retriever(namespace=namespace, k=5)
        docs = retriever_instance.invoke(query)
        if not docs:
            return ""
        formatted_chunks = []
        for d in docs:
            source = d.metadata.get("source", "Uploaded PDF")
            page = d.metadata.get("page", "?")
            formatted_chunks.append(
                f"[Source: {source}, Page: {page}]\n{d.page_content}"
            )
        return "\n\n".join(formatted_chunks)
    except Exception as e:
        return f"RAG_ERROR::{e}"

# --- Pydantic schemas for structured output ---
class RouteDecision(BaseModel):
    standalone_query: str = Field(
        ...,
        description="The contextually resolved, standalone query rephrased from conversation history (or unchanged if already standalone)."
    )
    route: Literal["rag", "web", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool = Field(..., description="True if retrieved information is sufficient to answer the user's question, False otherwise.")

# --- LLM instances with structured output where needed ---
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or "dummy-key"

router_llm = ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(RouteDecision)
judge_llm = ChatGroq(model=GROQ_MODEL, temperature=0).with_structured_output(RagJudge, method="json_mode")
answer_llm = ChatGroq(model=GROQ_MODEL, temperature=0.7)

from typing import List, Literal, TypedDict, Annotated
from langgraph.graph.message import add_messages

# --- Shared state type ---
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    standalone_query: str
    route: Literal["rag", "web", "answer", "end"]
    rag: str
    web: str

# --- Node 1: Unified Query Rewriter + Router (single LLM call) ---
def router_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering unified router_node (Rewrite + Route) ---")
    messages = state.get("messages", [])
    raw_query = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")

    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"Router received web search info: {web_search_enabled}")

    system_prompt = (
        "You are an intelligent query-rewriting and routing agent.\n"
        " Perform TWO tasks in a single structured output:\n"
        "1. `standalone_query`: Given the conversation history and the user's latest message, rephrase the latest message "
        "into a standalone query that resolves all pronouns/references (or return it unchanged if it is already standalone or a greeting).\n"
        "2. `route`: Direct the `standalone_query` to the most appropriate route. Prioritize using the **internal knowledge base (RAG)** "
        "for factual information likely contained in uploaded documents or well-established domain knowledge.\n"
    )

    if web_search_enabled:
        system_prompt += (
            "You **CAN** use web search for queries that require current, real-time, or live internet knowledge.\n"
            "\nChoose one of the following routes:\n"
            "- 'rag': Specific entities, uploaded document questions, procedures, technical facts, or general knowledge.\n"
            "- 'web': Current events, live data, recent news, or real-time information.\n"
        )
    else:
        system_prompt += (
            "**Web search is currently DISABLED.** You **MUST NOT** choose the 'web' route.\n"
            "\nChoose one of the following routes:\n"
            "- 'rag': Specific entities, uploaded document questions, procedures, or queries that would otherwise need search.\n"
        )

    system_prompt += (
        "- 'answer': Simple conversational or meta questions you can answer directly without lookup (e.g., 'What can you do?').\n"
        "- 'end': Pure greetings or small-talk where no factual lookup is needed (e.g., 'Hi', 'Thanks!'). If choosing 'end', you MUST populate `reply`.\n"
    )

    # Pass recent conversation history (up to last 6 messages) so follow-up questions are resolved in the same call
    history_window = messages[:-1][-6:] if len(messages) > 1 else []
    llm_messages = [("system", system_prompt)] + history_window + [("user", raw_query)]

    result: RouteDecision = router_llm.invoke(llm_messages)
    standalone_query = (result.standalone_query or raw_query).strip() or raw_query

    initial_router_decision = result.route
    router_override_reason = None

    if not web_search_enabled and result.route == "web":
        result.route = "rag"
        router_override_reason = "Web search disabled by user; redirected to RAG."
        print("Router decision overridden: changed from 'web' to 'rag' because web search is disabled.")

    print(f"Standalone query: {standalone_query} | Route: {result.route} | Reply: {result.reply}")

    out = {
        "messages": state["messages"],
        "standalone_query": standalone_query,
        "route": result.route,
    }
    if router_override_reason:
        out["initial_router_decision"] = initial_router_decision
        out["router_override_reason"] = router_override_reason

    if result.route == "end":
        out["messages"] = state["messages"] + [AIMessage(content=result.reply or "Hello! How can I help you today?")]

    print("--- Exiting unified router_node ---")
    return out

# --- Node 2: RAG lookup ---
def rag_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering rag_node ---")
    query = state.get("standalone_query", "")
    configurable = config.get("configurable", {})
    web_search_enabled = configurable.get("web_search_enabled", True)
    session_namespace = configurable.get("thread_id", "default") or "default"
    print(f"RAG node received web search info: {web_search_enabled}, namespace: {session_namespace}")
    print(f"RAG query: {query}")
    chunks = rag_search_tool.invoke({"query": query, "namespace": session_namespace})
    
    if chunks.startswith("RAG_ERROR::"):
        print(f"RAG Error: {chunks}. Checking web search enabled status.")
        # If RAG fails, and web search is enabled, try web. Otherwise, go to answer.
        next_route = "web" if web_search_enabled else "answer"
        return {**state, "rag": "", "route": next_route}

    if chunks:
        print(f"Retrieved RAG chunks (first 500 chars): {chunks[:500]}...")
    else:
        print("No RAG chunks retrieved.")

    judge_messages = [
        ("system", (
            "You are a judge evaluating if the **retrieved information** is **sufficient and relevant** "
            "to fully and accurately answer the user's question. "
            "Consider if the retrieved text directly addresses the question's core and provides enough detail."
            "If the information is incomplete, vague, outdated, or doesn't directly answer the question, it's NOT sufficient."
            "If it provides a clear, direct, and comprehensive answer, it IS sufficient."
            "If no relevant information was retrieved at all (e.g., 'No results found'), it is definitely NOT sufficient.\n\n"
            "You MUST return a JSON object with a single key 'sufficient' set to true or false. Do not include any other keys.\n"
            "JSON Format:\n"
            "{\n"
            "  \"sufficient\": true\n"
            "}"
        )),
        ("user", f"Question: {query}\n\nRetrieved info: {chunks}")
    ]
    verdict: RagJudge = judge_llm.invoke(judge_messages)
    print(f"RAG Judge verdict: {verdict.sufficient}")
    print("--- Exiting rag_node ---")
    
    # Decide next route based on sufficiency AND web_search_enabled
    if verdict.sufficient:
        next_route = "answer"
    else:
        next_route = "web" if web_search_enabled else "answer" # If not sufficient, only go to web if enabled
        print(f"RAG not sufficient. Web search enabled: {web_search_enabled}. Next route: {next_route}")

    return {
        **state,
        "rag": chunks,
        "route": next_route
    }

# --- Node 3: web search ---
def web_node(state: AgentState, config: RunnableConfig) -> AgentState:
    print("\n--- Entering web_node ---")
    query = state.get("standalone_query", "")
    
    web_search_enabled = config.get("configurable", {}).get("web_search_enabled", True)
    print(f"Web node received web search info: {web_search_enabled}")
    if not web_search_enabled:
        print("Web search node entered but web search is disabled. Skipping actual search.")
        return {**state, "web": "Web search was disabled by the user.", "route": "answer"}

    print(f"Web search query: {query}")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=3)]
        extracted = []
        for r in results:
            extracted.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", "")
            })
        snippets = "\n\n".join(f"Title: {item['title']}\nContent: {item['snippet']}" for item in extracted)
    except Exception as e:
        print(f"Web Error: {e}. Proceeding to answer with limited info.")
        snippets = ""

    print(f"Web snippets retrieved: {snippets[:200]}...")
    print("--- Exiting web_node ---")
    return {**state, "web": snippets, "route": "answer"}

# --- Node 4: final answer ---
def answer_node(state: AgentState) -> AgentState:
    print("\n--- Entering answer_node ---")
    user_q = state.get("standalone_query", "")
    
    ctx_parts = []
    if state.get("rag"):
        ctx_parts.append("Knowledge Base Information:\n" + state["rag"])
    if state.get("web"):
        # If web search was disabled, the 'web' field might contain a message like "Web search was disabled..."
        # We should only include actual search results here.
        if state["web"] and not state["web"].startswith("Web search was disabled"):
            ctx_parts.append("Web Search Results:\n" + state["web"])
    
    context = "\n\n".join(ctx_parts)
    if not context.strip():
        context = "No external context was available for this query. Try to answer based on general knowledge if possible."

    prompt = f"""Please answer the user's question using the provided context.
If the context is empty or irrelevant, try to answer based on your general knowledge.
When using information from the Knowledge Base, always cite the document source and page number inline (e.g., [Source: file.pdf, Page: 1]).

Question: {user_q}

Context:
{context}

Provide a helpful, accurate, and concise response based on the available information."""

    print(f"Prompt sent to answer_llm: {prompt[:500]}...")
    
    # Prepend conversation history to let LLM leverage context from past messages
    system_prompt = f"You are a helpful assistant. Use the following context if relevant:\n{context}"
    llm_messages = [("system", system_prompt)] + state["messages"][:-1] + [HumanMessage(content=prompt)]
    
    ans = answer_llm.invoke(llm_messages).content
    print(f"Final answer generated: {ans[:200]}...")
    print("--- Exiting answer_node ---")
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }

# --- Routing helpers ---
def from_router(st: AgentState) -> Literal["rag", "web", "answer", "end"]:
    return st["route"]

def after_rag(st: AgentState) -> Literal["answer", "web"]:
    return st["route"]

def after_web(_) -> Literal["answer"]:
    return "answer"

# --- Build graph ---
def build_agent(checkpointer):
    """Builds and compiles the LangGraph agent with a checkpointer."""
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("rag_lookup", rag_node)
    g.add_node("web_search", web_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("router")
    
    g.add_conditional_edges(
        "router",
        from_router,
        {
            "rag": "rag_lookup",
            "web": "web_search",
            "answer": "answer",
            "end": END
        }
    )
    
    g.add_conditional_edges(
        "rag_lookup",
        after_rag,
        {
            "answer": "answer",
            "web": "web_search"
        }
    )
    
    g.add_edge("web_search", "answer")
    g.add_edge("answer", END)

    agent = g.compile(checkpointer=checkpointer)
    return agent

# --- Global Execution Context Initialization (Serverless Warm Reuse) ---
checkpointer = MemorySaver()
rag_agent = build_agent(checkpointer)

def init_agent():
    """Idempotent initializer for backwards compatibility with FastAPI lifespan."""
    global checkpointer, rag_agent
    if rag_agent is None:
        checkpointer = MemorySaver()
        rag_agent = build_agent(checkpointer)

def close_agent():
    """No-op cleanup for stateless serverless execution."""
    pass