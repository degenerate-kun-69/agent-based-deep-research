import os
from dotenv import load_dotenv
import requests
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms import HuggingFaceHub
from langgraph.graph import END, StateGraph
from fpdf import FPDF

# Load .env variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# === Research Tool using Tavily ===
def search_web(query: str) -> str:
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5}
    )
    data = response.json()
    results = "\n".join([f"- {item['content']}" for item in data.get("results", [])])
    return results or "No results found."

# === Hugging Face Answer Generator ===
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5, "max_length": 300}, huggingfacehub_api_token=HF_TOKEN)

# === LangChain Tool ===
search_tool = Tool.from_function(
    name="web_search",
    func=search_web,
    description="Useful for online information gathering on any topic."
)

# === Agent: Research & Draft Answer ===
tools = [search_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# === LangGraph Workflow ===
def get_query(state):
    return {"query": input("🧠 Enter your research query: ")}

def research_and_answer(state):
    query = state["query"]
    response = agent.run(query)
    return {"query": query, "response": response}

def export_pdf(state):
    query, answer = state["query"], state["response"]
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Query:\n{query}\n\nAnswer:\n{answer}")
    pdf_path = "deep_research_result.pdf"
    pdf.output(pdf_path)
    print(f"✅ PDF saved at: {pdf_path}")
    return state

# === LangGraph Graph Definition ===
workflow = StateGraph()
workflow.add_node("query_input", get_query)
workflow.add_node("research", research_and_answer)
workflow.add_node("export", export_pdf)

workflow.set_entry_point("query_input")
workflow.add_edge("query_input", "research")
workflow.add_edge("research", "export")
workflow.add_edge("export", END)

app = workflow.compile()

# === Run the Agentic System ===
app.invoke({})