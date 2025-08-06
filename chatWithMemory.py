import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv
import os

# Load environment variables from .env file (make sure GROQ_API_KEY and SERPAPI_API_KEY set)
load_dotenv()

# --- Setup Tools ---
search = SerpAPIWrapper()
search_tool = Tool(
    name="Google Search",
    func=search.run,
    description="A tool to search the web using Google. Input should be a query string."
)
tools = [search_tool]

# --- Setup LLM ---
llm = ChatGroq(
    temperature=0,
    model="llama3-8b-8192",
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# --- Setup memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = st.session_state.memory

# --- Initialize Conversational ReAct Agent ---
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    max_iterations=3,
    verbose=True
)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Conversational ReAct Agent", layout="centered")
st.title("Conversational ReAct Agent with Memory")
st.info("Ask questions below, your conversation history will be remembered.")

# Initialize chat history in session_state if not present
if "chat" not in st.session_state:
    st.session_state.chat = []

# Handle Clear Chat before input widget is rendered
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

# --- Layout for input and buttons ---
input_col, send_col, clear_col = st.columns([8, 1, 1])

# Reset input BEFORE rendering input widget
if "clear_input" in st.session_state and st.session_state.clear_input:
    st.session_state["user_input"] = ""
    st.session_state.clear_input = False

with input_col:
    user_input = st.text_input(
        "Type your message",
        key="user_input",
        placeholder="Type a message...",
        label_visibility="collapsed",
    )

with send_col:
    send_clicked = st.button("Send", use_container_width=True)

with clear_col:
    clear_clicked = st.button("Clear Chat", use_container_width=True)

# Display chat bubbles
def display_chats():
    for message in st.session_state.chat:
        if message["role"] == "user":
            st.markdown(
                f"<div style='background:#005c4b; color:white; float:right; clear:both; padding:10px 18px; margin:6px 0; "
                f"border-radius:18px 18px 4px 18px; max-width:75%; text-align:right;'>{message['text']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#262d31; color:#dee1e6; float:left; clear:both; padding:10px 18px; margin:6px 0; "
                f"border-radius:18px 18px 18px 4px; max-width:75%;'>{message['text']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("<div style='clear:both;'></div>", unsafe_allow_html=True)

display_chats()

# Handle Send button
if send_clicked and user_input.strip() != "":
    st.session_state.chat.append({"role": "user", "text": user_input.strip()})

    try:
        response = agent.run(user_input.strip())
    except Exception as e:
        response = f"⚠️ Error from agent: {e}"

    st.session_state.chat.append({"role": "assistant", "text": response})

    # Set flag to clear input on rerun
    st.session_state.clear_input = True
    st.rerun()

# Handle Clear Chat button
if clear_clicked:
    st.session_state.chat = []
    memory.clear()
    st.session_state.clear_input = True
    st.rerun()
