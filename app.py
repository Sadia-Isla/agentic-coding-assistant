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

# --- Sidebar: User API Key ---
with st.sidebar:
    st.header("Settings")
    user_api_key = st.text_input("Enter Groq API Key", type="password")
    st.info("Get a free key from the [Groq Console](https://console.groq.com).")

# --- Tool Definition: The Python Sandbox ---
def execute_python_code(code: str) -> str:
    """Useful for running Python code to verify math, logic, or scripts."""
    output = io.StringIO()
    try:
        sys.stdout = output
        # Include math module in the execution scope
        exec(code, {"math": math, "print": print})
        sys.stdout = sys.__stdout__
        result = output.getvalue()
        return result if result else "Executed successfully (no output)."
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error: {str(e)}"

# --- Agent Initialization ---
if user_api_key:
    try:
        # 1. Initialize Groq LLM with a SUPPORTED model
        # llama-3.3-70b-versatile is currently the best performing free-tier model
        llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=user_api_key,
            temperature=0
        )

        # 2. Setup Tool
        tools = [
            Tool(
                name="python_repl",
                func=execute_python_code,
                description="Executes python code. Use this for math (e.g. square roots, primes)."
            )
        ]

        # 3. Pull the ReAct prompt
        prompt = hub.pull("hwchase17/react")

        # 4. Construct the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # 5. Create the executor
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            max_iterations=10
        )

        # --- Chat UI Logic ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Calculate the square root of 54321 and multiply by 12th prime..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Calculating..."):
                    try:
                        response = agent_executor.invoke({"input": user_query})
                        answer = response["output"]
                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Agent Error: {e}")

    except Exception as e:
        st.error(f"Initialization Error: {e}")
else:
    st.warning("Please enter your Groq API key in the sidebar to start.")
