# agent_graph.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any
import os
import re

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# ---- Small local model (very small & easy to run) ----
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---- Your retrieval tool (duck-typed adapter) ----
# Expect one of:
#   - function: search(query: str, k: int) -> List[Dict[str, Any]] with 'text' and optional 'source'
#   - class: RetrievalTool with .search(query: str, k: int)
try:
    import retrieval_tool as rt  # your file in project root
except Exception as e:
    rt = None
    print("[agent_graph] Warning: could not import retrieval_tool.py:", e)

# -----------------------------
# Config
# -----------------------------
HF_MODEL = os.environ.get("SMALL_HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))

CLINICAL_LLM_KEYWORDS = [
    # feel free to expand these as you use it
    r"\bclinical (?:llm|language model)s?\b",
    r"\btrial(?:s)?\b", r"\binclusion(?:/| and )?exclusion\b", r"\bie[ /-]?criteria\b",
    r"\bomop\b", r"\bohdsi\b", r"\bloinc\b", r"\bsnomed\b", r"\brxnorm\b",
    r"\bphenotyping\b", r"\bcohort\b", r"\bEHR\b", r"\bFHIR\b", r"\bCDISC\b",
    r"\bvalue ?set\b", r"\bconcept set\b", r"\bretrieval[- ]?augmented\b",
    r"\bclinicaltrials\.gov\b", r"\bML4H\b", r"\bCHIL\b", r"\bMLHC\b",
]

SYSTEM_RAG_PROMPT = """You are a helpful assistant for questions about *Clinical LLMs*.
Use the provided context to answer succinctly and cite sources inline like [S1], [S2].
If the context is insufficient, say so and explain what else you need."""

SYSTEM_GENERAL_PROMPT = """You are a concise, pragmatic assistant. Answer clearly and briefly."""

# -----------------------------
# Light heuristic classifier
# -----------------------------
def is_clinical_llm_question(text: str) -> bool:
    t = text.lower()
    for pat in CLINICAL_LLM_KEYWORDS:
        if re.search(pat, t):
            return True
    return False

# -----------------------------
# Retrieval adapter
# -----------------------------
def run_retrieval(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    if rt is None:
        return []
    # Try best-guess function/class names without forcing the user to rename their file
    if hasattr(rt, "search") and callable(rt.search):
        return rt.search(query=query, k=k)  # expected: list of {'text': ..., 'source': ...}
    if hasattr(rt, "RetrievalTool"):
        tool = rt.RetrievalTool()
        return tool.search(query=query, k=k)
    if hasattr(rt, "RetrievalClient"):
        client = rt.RetrievalClient()
        return client.search(query=query, k=k)
    # Fallback: nothing found
    return []

# -----------------------------
# Local generator with HF model
# -----------------------------
class LocalGenerator:
    def __init__(self, model_name: str = HF_MODEL):
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            do_sample=False,
            max_new_tokens=400,
            temperature=0.0,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"<|system|>\n{system_prompt}\n</s>\n<|user|>\n{user_prompt}\n</s>\n<|assistant|>\n"
        out = self.pipe(prompt)[0]["generated_text"]
        # Return only assistant segment after the last assistant tag
        if "<|assistant|>" in out:
            return out.split("<|assistant|>")[-1].strip()
        return out.strip()

GEN = LocalGenerator()

# -----------------------------
# LangGraph state
# -----------------------------
class ChatState(TypedDict):
    messages: List[Any]          # HumanMessage / AIMessage
    use_rag: bool
    retrieved: List[Dict[str, Any]]

# -----------------------------
# Nodes
# -----------------------------
def classify_intent(state: ChatState) -> ChatState:
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    query = last_user.content if last_user else ""
    state["use_rag"] = is_clinical_llm_question(query)
    return state

def retrieve(state: ChatState) -> ChatState:
    if not state.get("use_rag"):
        state["retrieved"] = []
        return state
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    query = last_user.content if last_user else ""
    docs = run_retrieval(query, k=TOP_K)
    state["retrieved"] = docs
    return state

def generate(state: ChatState) -> ChatState:
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_q = last_user.content if last_user else ""

    if state.get("use_rag"):
        # Build a compact context with inline source handles
        ctx_lines = []
        for i, d in enumerate(state.get("retrieved", []), start=1):
            src = d.get("source") or d.get("id") or f"S{i}"
            snippet = d.get("text", "").strip()
            ctx_lines.append(f"[S{i}] ({src}) {snippet}")
        context = "\n".join(ctx_lines) if ctx_lines else "No relevant documents retrieved."
        full_user = f"Context:\n{context}\n\nQuestion:\n{user_q}\n\nInstructions: Cite sources like [S1]."
        system = SYSTEM_RAG_PROMPT
    else:
        full_user = user_q
        system = SYSTEM_GENERAL_PROMPT

    answer = GEN.chat(system_prompt=system, user_prompt=full_user)
    state["messages"].append(AIMessage(content=answer))
    return state

# -----------------------------
# Graph wiring
# -----------------------------
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.add_edge(START, "classify_intent")
    # branch: if use_rag → retrieve → generate; else → generate
    def should_retrieve(state: ChatState) -> str:
        return "retrieve" if state.get("use_rag") else "generate"

    g.add_conditional_edges("classify_intent", should_retrieve, {"retrieve": "retrieve", "generate": "generate"})
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()

# -----------------------------
# Simple REPL for local testing
# -----------------------------
def run_chat():
    app = build_graph()
    state: ChatState = {"messages": [], "use_rag": False, "retrieved": []}
    print("RAG Chatbot ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        state["messages"].append(HumanMessage(content=q))
        state = app.invoke(state)
        print("\nAssistant:", state["messages"][-1].content)

if __name__ == "__main__":
    run_chat()
