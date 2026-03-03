💻 Agentic Python Coding Assistant
An autonomous AI Coding Agent built with Streamlit and LangChain. This assistant doesn't just write code—it executes and tests it in a secure Python sandbox to verify its own logic before giving you the final answer.

🚀 Overview
Traditional LLMs often hallucinate math or complex logic. This "Agentic" assistant uses the ReAct (Reason + Act) framework to:
Think about the user's request.
Write a Python script to solve it.
Execute that script using a built-in Python REPL tool.
Observe the output or errors.
Refine and repeat until the correct result is achieved.

🛠️ Tech Stack
Frontend: Streamlit (for a clean, interactive Chat UI).
Orchestration: LangChain (using the ReAct Agent logic).
LLM Provider: Groq Cloud (utilizing Llama-3.3-70b-versatile for high-speed reasoning).
Execution: Custom Python Sandbox (using io and sys for safe stdout capture).

✨ Key Features
Self-Correcting: If the generated code throws an error, the agent reads the traceback and fixes the code automatically.
Math & Logic Specialist: Perfect for calculating primes, square roots, or complex data manipulations that standard LLMs struggle with.

Secure Execution: Runs code in a controlled environment within the session.
Session Memory: Maintains chat history using Streamlit's session_state.

⚙️ Setup & Installation
Clone the repo:
bash
git clone https://github.com
cd agentic-coding-assistant
Use code with caution.

Create a Virtual Environment:
bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Mac/Linux
Use code with caution.

Install Dependencies:
bash
pip install -r requirements.txt
Use code with caution.

Run the App:
bash
streamlit run app.py
Use code with caution.

