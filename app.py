import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
import sys
import io

# --- Page Config ---
st.set_page_config(page_title="Agentic Coder", page_icon="🤖")
st.title("🤖 Agentic Coder")
st.caption("An autonomous AI that writes and tests Python code.")

# --- API Key Handling ---
# This allows users to use their own key (Safe for your wallet!)
with st.sidebar:
    user_api_key = st.text_input("Enter Google API Key", type="password")
    "[Get a free Gemini API key](https://aistudio.google.com/app/apikey)"

# --- Custom Tool: Python Executor ---
def execute_python_code(code: str) -> str:
    output = io.StringIO()
    try:
        sys.stdout = output
        exec(code)
        sys.stdout = sys.__stdout__
        return output.getvalue() or "Executed successfully (No output)."
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {str(e)}"

# --- Agent Logic ---
if user_api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=user_api_key)
    tools = [Tool(name="Python_REPL", func=execute_python_code, description="Run python code")]
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # This captures the "Thinking" process and shows it in the UI
            st_callback = st.container() 
            response = agent.run(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("Please add your Gemini API key in the sidebar to begin.")
