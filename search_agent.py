import time
from typing import TypedDict, List
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 3

class SearchAgent(TypedDict):
    question: str
    tool_result: str
    search_failed: bool

search_tavily = TavilySearch(max_results = 3)

def tool_search(state: SearchAgent):
    question = state["question"]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = search_tavily.invoke(question)
            #Tavily reports some failures (bad key, quota) as an "error" field instead of raising
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(response["error"])
            return{**state, "tool_result": response, "search_failed": False}
        except Exception as e:
            print(f"[search] attempt {attempt}/{MAX_ATTEMPTS} failed: {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))  #1s, 2s backoff
    #Let the writer fall back to its own knowledge instead of failing the whole request
    return{**state, "tool_result": "", "search_failed": True}

workflow = StateGraph(SearchAgent)
workflow.set_entry_point("tool_search")
workflow.add_node("tool_search", tool_search)
workflow.add_edge("tool_search", END)

search_app = workflow.compile()
