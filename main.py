import os
from dotenv import load_dotenv
import requests
from fpdf import FPDF
from typing import TypedDict

#new template for llm
"""
# Initialize Hugging Face model using updated class and pass parameters explicitly

research_llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",  # Your model of choice
    temperature=0.5,  # Pass temperature directly
    max_new_tokens=300,  # Pass max_new_tokens directly
    huggingfacehub_api_token=HF_TOKEN  # Provide the API token
)

"""
# Updated LangChain imports
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.llms import HuggingFaceHub -------- Deprecated
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# ======================
# State Schema Definition
# ======================

class ResearchState(TypedDict):
    query: str
    research_data: str
    final_answer: str

# ======================
# Research Agent System
# ======================

def tavily_search(query: str) -> str:
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 7,
                "search_depth": "advanced"
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return f"Research Summary:\n{data.get('answer', '')}\n\nSources:\n" + "\n".join(
            [f"- {res['url']}" for res in data.get('results', [])[:5]]
        )
    except Exception as e:
        return f"Research error: {str(e)}"

# Initialize HF models with updated parameters
research_llm = HuggingFaceEndpoint( ##switch to HuggingFaceEndpoint because HuggingFaceHub is deprecated with updated syntax
    repo_id="google/zephyr-7b-beta",
    temperature=0.2,
    max_new_tokens=1024,
    top_k=50,
    top_p=0.8,
    huggingfacehub_api_token=HF_TOKEN
)

drafting_llm = HuggingFaceEndpoint( ##switch to HuggingFaceEndpoint because HuggingFaceHub is deprecated with updated syntax
    repo_id="HuggingFaceH4/zephyr-7b-beta",
        temperature= 0.3,
        max_new_tokens= 1024,
        top_k= 50,
        top_p= 0.8,
    huggingfacehub_api_token=HF_TOKEN
)

# Research Agent Setup
research_tools = [
    Tool(
        name="DeepWebSearch",
        func=tavily_search,
        description="Comprehensive web search for in-depth research"
    )
]


#research_prompt = PromptTemplate.from_template(
#    """[System] You are an AI research analyst. Investigate: {query}
#    
#   Steps:
#    1. Use DeepWebSearch tool
#    2. Analyze multiple perspectives
#    3. Identify key facts with sources
#   
#   [Assistant] Let's research this step-by-step:"""
#)


# Updated research prompt with tools and agent scratchpad to fix error
research_prompt = PromptTemplate.from_template(
    """[System] You are an AI research analyst. Conduct thorough investigation of:
"{query}"

Follow these steps:
1. Use DeepWebSearch to gather information
2. Analyze multiple perspectives
3. Identify key facts and sources
4. Prepare raw research data

[Tools] {tools}  # Include the tools variable here
[Tool Names] {tool_names}  # Include the tool_names variable here
[Agent Scratchpad] {agent_scratchpad}  # Include the agent_scratchpad variable here

[Assistant] Let's research this step-by-step. First, I'll use the DeepWebSearch tool."""
)


research_agent = create_react_agent(research_llm, research_tools, research_prompt)
research_executor = AgentExecutor(agent=research_agent, tools=research_tools, verbose=True)

# ======================
# Drafting Agent System
# ======================

drafting_prompt = PromptTemplate.from_template(
    """[System] Create a professional report from this research:
    
    Topic: {query}
    Data: {research_data}
    
    Structure:
    1. Executive Summary
    2. Key Findings with Sources
    3. Analysis
    4. Conclusions
    
    [Assistant] Here's the structured report:"""
)

drafting_chain = drafting_prompt | drafting_llm

# =====================
# PDF Export Function
# =====================

def export_to_pdf(state: ResearchState):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Title
        pdf.cell(0, 10, f"Research Report: {state['query']}", ln=1)
        pdf.ln(10)
        
        # Content
        pdf.multi_cell(0, 8, state['final_answer'])
        
        pdf_path = "research_report.pdf"
        pdf.output(pdf_path)
        print(f"PDF saved to: {pdf_path}")
        return state
    except Exception as e:
        print(f"PDF error: {str(e)}")
        return state

# =====================
# LangGraph Workflow
# =====================
workflow = StateGraph(state_schema=ResearchState) #fixed a syntax error


def input_query(state: ResearchState):
    return {"query": input("Enter research topic: ")}

def conduct_research(state: ResearchState):
    result = research_executor.invoke({"input": state["query"]})
    return {"research_data": result["output"]}

def draft_answer(state: ResearchState):
    response = drafting_chain.invoke({
        "query": state["query"],
        "research_data": state["research_data"]
    })
    return {"final_answer": response}

# Build workflow
workflow.add_node("input_query", input_query)
workflow.add_node("conduct_research", conduct_research)
workflow.add_node("draft_answer", draft_answer)
workflow.add_node("export_pdf", export_to_pdf)

workflow.set_entry_point("input_query")
workflow.add_edge("input_query", "conduct_research")
workflow.add_edge("conduct_research", "draft_answer")
workflow.add_edge("draft_answer", "export_pdf")
workflow.add_edge("export_pdf", END)

app = workflow.compile()

# Run the system
if __name__ == "__main__":
    app.invoke({"query": ""})