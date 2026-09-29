from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class WriterAgent(TypedDict):
    question: str
    documents: List[Document]
    is_retrieval: str
    chat_history: List
    tool_result: str
    answer: str
    hallucinate_count: int
    rewritten: str
    
llm = ChatOpenAI(model = "gpt-4o", temperature = 0)

def generate(state: WriterAgent):
    question = state["question"]
    is_retrieval = state["is_retrieval"].lower()
    #print("DEBUG generate is_retrieval:", repr(is_retrieval))
    tool_result = str(state["tool_result"])
    #print("DEBUG tool_result in writer:", tool_result[:300])
    chat_history = state["chat_history"]
    context = ""
    for doc in state["documents"]:
        context = context + doc.page_content
    

    
    if "search" in is_retrieval:
        prompt = f"""Generate a response using this content from the web as your information source.
        question: {question}
        information: {tool_result}
        Answer: """
    elif "retrieve" in is_retrieval:
        prompt = f"""Generate a response using the document content below.
        The documents may contain raw table data — do your best to interpret it.
        If you can see numbers that likely answer the question, use them.
        Only say "I don't know" if the documents contain absolutely nothing related.

        Question: {question}
        Documents: {context}
        Answer:"""
    else:
        prompt = f"""Generate a response using your knowledge and chat history.
        question: {question}
        chat history: {chat_history}
        Answer: """
        
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}

def hallucination_check(state: WriterAgent) -> WriterAgent:
    answer = state["answer"]
    documents = state["documents"]
    hallucination_count = state["hallucinate_count"]
    is_retrieval = state["is_retrieval"].lower()
    context=""
    for doc in documents:
        context = context + doc.page_content
    if not context or "retrieve" not in is_retrieval:
        return{**state, "hallucinate_count": 0}
    else:
        prompt = f""" Does this answer accuratley reflect whatever the documents gave?
        answer: {answer}
        documents: {context}
        Answer only with YES or NO
        """
        response = llm.invoke(prompt)
        
    if "yes" in response.content.lower():
        return{**state, "hallucinate_count": hallucination_count}
    else:
        return{**state, "hallucinate_count": hallucination_count + 1}
    
def should_hallucinate(state: WriterAgent) -> str:
    if state["hallucinate_count"] == 0:
        return "pass"
    elif state["hallucinate_count"] < 2:
        return "regenerate"
    else:
        return "pass"
    
workflow = StateGraph(WriterAgent)
workflow.set_entry_point("generate")
workflow.add_node("generate", generate)
workflow.add_node("hallucination_check", hallucination_check)
workflow.add_edge("generate", "hallucination_check")
workflow.add_conditional_edges(
    "hallucination_check",
    should_hallucinate, {
        "pass": END,
        "regenerate": "generate"
    }
    )

writer_app = workflow.compile()
