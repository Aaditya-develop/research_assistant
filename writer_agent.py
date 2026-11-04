from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_ATTEMPTS = 2

class WriterAgent(TypedDict):
    question: str
    documents: List[Document]
    is_retrieval: str
    chat_history: List
    answer: str
    hallucinate_count: int
    grounded: bool

llm = ChatOpenAI(model = "gpt-4o", temperature = 0, timeout = 60, max_retries = 3)


def format_context(documents: List[Document]) -> str:
    #Separate chunks and label their page so the model doesn't blend numbers across chunks
    parts = []
    for i, doc in enumerate(documents):
        parts.append(f"[Chunk {i+1} | page {doc.metadata.get('page', '?')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(chat_history: List) -> str:
    lines = []
    for turn in chat_history:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines) if lines else "(none)"


def generate(state: WriterAgent):
    question = state["question"]
    is_retrieval = state["is_retrieval"].lower()
    history = format_history(state["chat_history"])
    context = format_context(state["documents"])

    if "retrieve" in is_retrieval:
        prompt = f"""Generate a response using the document content below, taken from Apple's fiscal 2024 Form 10-K.
        The documents may contain raw table data — do your best to interpret it.
        If you can see numbers that likely answer the question, use them.
        Only say "I don't know" if the documents contain absolutely nothing related.

        Chat history: {history}
        Question: {question}
        Documents: {context}
        Answer:"""
    else:
        prompt = f"""Generate a response using your knowledge and chat history.
        Chat history: {history}
        Question: {question}
        Answer: """

    #On a retry, tell the model why its last answer was rejected, otherwise temperature 0 just repeats it
    if state.get("hallucinate_count", 0) > 0 and state.get("answer"):
        prompt += f"""

        Your previous answer was rejected because it was not supported by the documents:
        {state["answer"]}
        Write a new answer that only states facts and numbers that appear in the documents.
        If the documents don't contain the answer, say so."""

    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


def hallucination_check(state: WriterAgent) -> WriterAgent:
    answer = state["answer"]
    documents = state["documents"]
    hallucination_count = state["hallucinate_count"]
    is_retrieval = state["is_retrieval"].lower()
    context = format_context(documents)
    if not documents or "retrieve" not in is_retrieval:
        return{**state, "hallucinate_count": 0, "grounded": True}

    prompt = f""" Does this answer accurately reflect whatever the documents gave?
    answer: {answer}
    documents: {context}
    Answer only with YES or NO
    """
    response = llm.invoke(prompt)

    if response.content.strip().lower().startswith("yes"):
        #Passed: reset the count so should_hallucinate ends the loop
        return{**state, "hallucinate_count": 0, "grounded": True}
    else:
        return{**state, "hallucinate_count": hallucination_count + 1, "grounded": False}

def should_hallucinate(state: WriterAgent) -> str:
    if state["hallucinate_count"] == 0:
        return "pass"
    elif state["hallucinate_count"] < MAX_ATTEMPTS:
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
