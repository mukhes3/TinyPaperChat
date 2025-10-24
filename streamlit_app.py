# streamlit_app.py
import streamlit as st
from langchain_core.messages import HumanMessage
from agent_graph import build_graph, ChatState

st.set_page_config(page_title="Local RAG Chat (LangGraph)", layout="centered")

# --- Initialize persistent objects ---
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "state" not in st.session_state:
    # RAG always on
    st.session_state.state = ChatState(messages=[], use_rag=True, retrieved=[])
else:
    # Ensure legacy sessions have RAG on
    if not st.session_state.state.get("use_rag", False):
        st.session_state.state["use_rag"] = True

# --- Header + input ---
st.title("🧪 Local RAG Chat (LangGraph) · 🧠 RAG: ON")
user_q = st.chat_input("Ask me something...")

# --- Render history ---
for msg in st.session_state.state["messages"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Handle new input ---
if user_q:
    st.session_state.state["messages"].append(HumanMessage(content=user_q))
    st.session_state.state = st.session_state.graph.invoke(st.session_state.state)

    # Display the assistant's latest message (assumes graph appended an AIMessage)
    with st.chat_message("assistant"):
        st.markdown(st.session_state.state["messages"][-1].content)

