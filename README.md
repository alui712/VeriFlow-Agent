# VeriFlow-Agent: Self-Correcting RAG System

## 🚀 Project Overview
VeriFlow is an advanced **Agentic RAG (Retrieval-Augmented Generation)** system designed to solve the "hallucination" problem in LLMs. unlike standard RAG pipelines that blindly trust retrieved documents, VeriFlow implements a **Graph-based State Machine** that evaluates its own answers before showing them to the user.

If the answer is factually unsupported, the agent autonomously rewrites the search query and loops back to find better information—mimicking human research behavior.

## ⚙️ Architecture (LangGraph)
The system operates on a cyclic graph with the following nodes:
1.  **Retrieve:** Fetches real-time data using the **Tavily Search API**.
2.  **Generate:** Synthesizes an answer using **GPT-4o-mini**.
3.  **Critique (The Judge):** A dedicated LLM prompts checks if the generated answer is grounded in the retrieved documents.
4.  **Refine (Loop):** If the critique fails, the agent enters a feedback loop, rewriting the query and searching again.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (StateGraph architecture)
* **LLM Framework:** LangChain
* **Models:** OpenAI GPT-4o-mini
* **Search Tool:** Tavily AI Search
* **Validation:** Pydantic (Structured Output)

## ⚡ How to Run
1.  Clone the repo:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/VeriFlow-Agent.git](https://github.com/YOUR_USERNAME/VeriFlow-Agent.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Add your API keys to a `.env` file.
4.  Run the agent:
    ```bash
    python main.py
    ```
