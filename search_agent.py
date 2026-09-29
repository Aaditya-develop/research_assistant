from typing import TypedDict, List
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

class SearchAgent(TypedDict):
    question: str
    tool_result: str
    
search_tavily = TavilySearch(max_results = 3)

def tool_search(state: SearchAgent):
    question = state["question"]
    response = search_tavily.invoke(question)
    return{**state, "tool_result": response}

workflow = StateGraph(SearchAgent)
workflow.set_entry_point("tool_search")
workflow.add_node("tool_search", tool_search)
workflow.add_edge("tool_search", END)

search_app = workflow.compile()