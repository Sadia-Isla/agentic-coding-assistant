import streamlit as st
import sys
import io
import math
from langchain_groq import ChatGroq
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub 
from langchain_core.tools import Tool

# --- Page Setup ---
st.set_page_config(page_title="Agentic Coder", page_icon="⚡")
st.title("⚡ Agentic Coder")
st.markdown("An autonomous agent powered by Llama 3.3.")

# --- Sidebar: Settings & Actions ---
with st.sidebar:
    st.header("Settings")
    user_api_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get a free key from the [Groq Console](https://console.groq.com).")
    
    st.markdown("---")
    # 🆕 Clear Chat History Button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Tool Definition: The Python Sandbox ---
def execute_python_code(code: str) -> str:
    """
    Executes python code. Input must be valid python code. 
    You MUST use print() statements to see any output.
    """
    code = code.strip().strip('```python').strip('```')
    output = io.StringIO()
    try:
        sys.stdout = output
        exec(code, {"math": math, "print": print})
        sys.stdout = sys.__stdout__
        result = output.getvalue()
        return result if result else "Success: Code executed. Remember to use print() to see results."
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {str(e)}"

# --- Agent Initialization ---
if user_api_key:
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=user_api_key,
            temperature=0
        )

        tools = [
            Tool(
                name="python_repl",
                func=execute_python_code,
                description="Use this to run Python code. Input should be the code string. Always print() your final result."
            )
        ]

        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=15
        )

        # --- Chat UI Logic ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask me to write code or solve a math problem..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Process request using LangChain [AgentExecutor](https://python.langchain.com)
                        response = agent_executor.invoke({"input": user_query})
                        answer = response["output"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Agent error: {e}")

    except Exception as e:
        st.error(f"Initialization Error: {e}")
else:
    st.warning("Please enter your Groq API key in the sidebar to start.")
