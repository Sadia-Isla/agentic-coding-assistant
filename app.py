import streamlit as st
import sys
import io
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

# --- Page Setup ---
st.set_page_config(page_title="Agentic Coder", page_icon="💻")
st.title("💻 Agentic Coder")
st.markdown("An autonomous Python agent that writes and tests its own code.")

# --- Sidebar: User API Key ---
with st.sidebar:
    st.header("Settings")
    user_api_key = st.text_input("Enter Gemini API Key", type="password")
    st.info("Your key is used only for this session and is not stored.")
    st.markdown("[Get a free API key here](https://aistudio.google.com/app/apikey)")

# --- Tool Definition: The Python Sandbox ---
def execute_python_code(code: str) -> str:
    """Useful for running Python code to verify logic or math."""
    output = io.StringIO()
    try:
        sys.stdout = output
        # Use a dictionary for local/global scope
        exec(code, {})
        sys.stdout = sys.__stdout__
        result = output.getvalue()
        return result if result else "Executed successfully (no output)."
    except Exception as e:
        sys.stdout = sys.__stdout__
        return f"Error encountered: {str(e)}"

# --- Agent Initialization ---
if user_api_key:
    try:
        # 1. Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=user_api_key,
            temperature=0
        )

        # 2. Setup Tool
        tools = [
            Tool(
                name="python_repl",
                func=execute_python_code,
                description="Executes python code. Input should be valid python code."
            )
        ]

        # 3. Pull the standard ReAct prompt from LangChain Hub
        prompt = hub.pull("hwchase17/react")

        # 4. Construct the ReAct agent
        agent = create_react_agent(llm, tools, prompt)

        # 5. Create the executor
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )

        # --- Chat UI Logic ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if user_query := st.chat_input("Ask me to write and run a script..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking and coding..."):
                    # Use .invoke() as per the [LangChain Runnable Interface](https://python.langchain.com)
                    response = agent_executor.invoke({"input": user_query})
                    answer = response["output"]
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"Initialization Error: {e}")
else:
    st.warning("Please enter your Gemini API key in the sidebar to start.")
