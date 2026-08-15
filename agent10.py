from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from retriever_agent import retriever_app
from search_agent import search_app
from writer_agent import writer_app
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler()


class OrchestratorState(TypedDict):
    question: str
    documents: List[Document]
    tool_result: str
    is_retrieval: str
    chat_history: List
    answer: str
    hallucinate_count: int
    
llm = ChatOpenAI(model = "gpt-4o", temperature = 0)

def decide(state: OrchestratorState) -> OrchestratorState:
    question = state["question"]
    chat_history = state["chat_history"]
    prompt = f"""You are a router. Decide how to answer this question.

    You have access to a vectorstore containing SEC 10-K filings.

    Rules:
    - If the question is about company financials, annual reports, risk factors, 
    revenue, business operations, or SEC filings → RETRIEVE
    - If the question needs current news or live information → SEARCH
    - If the question is general knowledge or conversational → GENERATE

    Question: {question}
    Chat history: {chat_history}

    Respond with ONE word only: RETRIEVE, SEARCH, or GENERATE
    """
    response = llm.invoke(prompt)
    # print("DEBUG decide:", repr(response.content))
    return{**state, "is_retrieval": response.content}

def should_route(state: OrchestratorState) -> str:
    if state["is_retrieval"].lower() == "retrieve":
        return "retrieve"
    elif state["is_retrieval"].lower() == "search":
        return "search"
    else:
        return "generate"

def run_retriever(state: OrchestratorState) -> OrchestratorState:
    result = retriever_app.invoke({
        "question": state["question"],
        "documents": []
    })
    return{**state,
           "documents": result["documents"],
           }

def run_search(state: OrchestratorState) -> OrchestratorState:
    result = search_app.invoke({
        "question": state["question"],
        "tool_result": state["tool_result"]
    })
    # print("DEBUG run_search result:", type(result["tool_result"]), str(result["tool_result"])[:200])
    return{**state,"tool_result": result["tool_result"]}

def run_writer(state: OrchestratorState) -> OrchestratorState:
    result = writer_app.invoke({
        "question": state["question"],
        "documents": state["documents"],
        "is_retrieval": state["is_retrieval"],
        "chat_history": state["chat_history"],
        "tool_result": state["tool_result"],
        "answer": state["answer"],
        "hallucinate_count": state["hallucinate_count"]
    })
    return {**state,
            "answer": result["answer"],
            "chat_history": result["chat_history"]
            }

workflow = StateGraph(OrchestratorState)
workflow.set_entry_point("decide")
workflow.add_node("decide", decide)
workflow.add_node("run_retriever", run_retriever)
workflow.add_node("run_search", run_search)
workflow.add_node("run_writer", run_writer)
workflow.add_conditional_edges(
    "decide",
    should_route,
    {
        "search": "run_search",
        "retrieve": "run_retriever",
        "generate": "run_writer"
    }
)
workflow.add_edge("run_search", "run_writer")
workflow.add_edge("run_retriever", "run_writer")
workflow.add_edge("run_writer", END)

orchestrator_app = workflow.compile()

if __name__ == "__main__":
    chat_history = []
    while True:
        question = input("You: ")
        if "quit" in question.lower():
            break
        result = orchestrator_app.invoke({
            "question": question,
            "documents": [],
            "tool_result": "",
            "is_retrieval": "",
            "chat_history": chat_history,
            "answer": "",
            "hallucinate_count": 0,
        },
        config={"callbacks": [langfuse_handler]})
        
        chat_history = result["chat_history"]
        print("Agent:", result["answer"])
        print()
    