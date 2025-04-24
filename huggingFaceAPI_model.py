import os
from dotenv import load_dotenv
import requests
from fpdf import FPDF
from typing import TypedDict

# Updated LangChain imports with output parser for fixing output issues to pdf format
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_community.llms import HuggingFaceHub -------- Deprecated
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

# ======================
# Initialize HF models with updated parameters
# ======================
research_llm = HuggingFaceEndpoint( ##switch to HuggingFaceEndpoint because HuggingFaceHub is deprecated with updated syntax
    repo_id="HuggingFaceH4/zephyr-7b-beta", # fixed accidental repo_id call to google
    temperature=0.2,
    max_new_tokens=512, #reduced tokens to 512 to avoid long outputs
    top_k=50,
    top_p=0.8,
    huggingfacehub_api_token=HF_TOKEN,
)

drafting_llm = HuggingFaceEndpoint( ##switch to HuggingFaceEndpoint because HuggingFaceHub is deprecated with updated syntax
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    temperature= 0.3,
    max_new_tokens= 512, #reduced tokens to 512 to avoid long outputs
    top_k= 50,
    top_p= 0.8,
    huggingfacehub_api_token=HF_TOKEN,
)

# Research Agent Setup
research_tools = [
    Tool(
        name="DeepWebSearch",
        func=tavily_search,
        description="Comprehensive web search for in-depth research"
    )
]


# Updated research prompt with tools and agent scratchpad to fix error
research_prompt = PromptTemplate.from_template( # fixed prompt template to use the correct format
    """[System] You are an AI research analyst. Your job is to thoroughly investigate and answer the user query using appropriate tools.

You must always respond using this format:
Thought: <your reasoning about which tool to use>
Action: <tool_name>
Action Input: <input for the selected tool>

Example:
Thought: The query is about future trends, so a search engine is appropriate.
Action: WebSearch
Action Input: "next possible stock shortage"

Follow these steps for every query:
1. Use the appropriate research tool (e.g., DeepWebSearch)
2. Analyze information from multiple perspectives
3. Identify and extract key facts and credible sources
4. Provide raw research data to the next agent

[Tools] {tools}
[Tool Names] {tool_names}
[Agent Scratchpad] {agent_scratchpad}

Question: {query}

[Assistant] Let's research this step-by-step. First, I'll decide the best tool to begin with."""
)


research_agent = create_react_agent(research_llm, research_tools, research_prompt)
research_executor = AgentExecutor(agent=research_agent, tools=research_tools, verbose=True)

# ======================
# Drafting Agent System
# ======================

drafting_prompt = PromptTemplate.from_template(
    """[System] You are an AI writing assistant. Your task is to generate a well-structured and professional report based on the research data provided.

Topic: {query}

Raw Research Data:
{research_data}

Guidelines:
- Maintain a formal tone suitable for professional or academic use.
- Summarize and analyze key insights accurately.
- Cite important sources when referencing facts.
- Ensure clarity and logical flow in each section.

Structure:
1. Executive Summary
2. Key Findings (with brief source mentions)
3. Analysis & Insights
4. Conclusion & Recommendations

[Assistant] Here's the structured report:"""
)

drafting_chain = drafting_prompt | drafting_llm

# =====================
# PDF Export Function
# =====================

def export_to_pdf(state: ResearchState):
    try:
        output_dir = "output" #send pdf to output folder
        os.makedirs(output_dir, exist_ok=True) #ensure directory exists
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

# fixed input_query function to use state["query"] instead of input()
"""
def input_query(state: ResearchState):
    return {"query": input("Enter research topic: ")}
"""
def input_query(state: ResearchState):
    return {"query": state["query"]}

# Manually use the output parser after getting model output
def parse_output(model_output: str) -> dict:
    try:
        # Attempt ReAct-style parsing
        return ReActSingleInputOutputParser().parse(model_output)
    except Exception as e:
        # If ReAct parsing fails, log the error and return a default action
        print(f"ReAct parsing failed with error: {e}")
        return {
            "thought": "LLM returned full answer",
            "action": "FinalAnswer",
            "action_input": model_output.strip()
        }
    

def conduct_research(state: ResearchState):
    result = research_executor.invoke({"query": state["query"]}) #fixed another syntax error for input
    # fallback if "output" key is missing
    output = result.get("output") if isinstance(result, dict) else str(result)
    # apply output parser to the result
    parsed_result = parse_output(result["output"])
    return {"research_data": parsed_result["action_input"]} #send parsed output 

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
    user_query = input("Enter your research topic: ")# forgot to add user input for query XD
    app.invoke({"query": user_query})