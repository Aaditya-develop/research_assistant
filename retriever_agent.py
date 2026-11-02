import re
import threading
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from vectorstore import VectorStore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model = "gpt-4o", temperature = 0, timeout = 30, max_retries = 3)
vs = VectorStore()

#Cross-encoder scores: real matches score above +1, vague questions below -2, unrelated ones around -10.
#If the best chunk scores below this, rewrite the question and search again.
RELEVANCE_THRESHOLD = 0.0
MAX_REWRITES = 1
TOP_K = 8
KEEP_PER_SEARCH = 2

class RetrieveAgent(TypedDict):
    question: str
    documents: List[Document]
    search_query: str
    top_score: float
    rewrite_count: int


#Question words like "what" and "does" are rare in a 10-K, so BM25 weights them higher than the actual topic
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "had", "has", "have",
    "how", "i", "in", "is", "it", "its", "me", "much", "many", "of", "on", "or", "so", "tell", "that", "the",
    "their", "them", "there", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your", "apple", "apple's", "company", "s", "10", "k", "form",
}

#Page headers like "Apple Inc. | 2024 Form 10-K | 6" are on many chunks and say nothing about the content,
#but they make any query mentioning "10-K" or "2024" match those chunks. Ignore them when scoring.
PAGE_HEADER = re.compile(r"^#*\s*[^|\n]+\|\s*\d{4} Form 10-K(\s*\|\s*\d+)?\s*$", re.MULTILINE)
FILING_REFERENCE = re.compile(r"\b(in|from|according to)?\s*(its|their|the|apple'?s)?\s*((fiscal )?(year )?2024 )?(form )?10-k( filing)?( for fiscal (year )?2024)?\b", re.IGNORECASE)

def scoring_text(text: str) -> str:
    return PAGE_HEADER.sub("", text).strip()

def scoring_query(query: str) -> str:
    #"...disclosed in its 2024 10-K?" → "...disclosed?" — every chunk is from that filing anyway
    cleaned = re.sub(r"\s+", " ", FILING_REFERENCE.sub(" ", query)).strip()
    return cleaned or query

def tokenize(text: str) -> List[str]:
    #Lowercase and strip punctuation so "Revenue," and "revenue" match in BM25
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    #Building BM25 over every chunk is slow, so build it once and only rebuild when the collection changes
    def __init__(self):
        self.lock = threading.Lock()
        self.count = -1
        self.docs = []
        self.bm25 = None

    def get(self):
        with self.lock:
            count = vs.count()
            if count != self.count:
                self.docs = vs.get_all_documents()
                self.bm25 = BM25Okapi([tokenize(scoring_text(doc.page_content)) for doc in self.docs]) if self.docs else None
                self.count = count
            return self.bm25, self.docs

bm25_index = BM25Index()


def hybrid_search(query: str, rerank_queries: List[str], extra_docs: List[Document] = ()) -> List[tuple]:
    #BM25 search
    bm25, all_docs = bm25_index.get()
    if bm25 is None:
        return []
    bm25_scores = bm25.get_scores(tokenize(scoring_query(query)))
    bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]
    bm25_docs = [all_docs[i] for i in bm25_top]

    #Semantic search
    sem_docs = [doc for doc, distance in vs.search_with_scores(query, k = TOP_K)]

    #Merge both (plus any earlier results), skipping chunks we've already seen
    seen = set()
    merged_list = []
    for doc in list(extra_docs) + sem_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            merged_list.append(doc)

    #Reranker: score each chunk against every query and keep its best score, highest first.
    #After a rewrite this means the user's own question and the rewrite both count.
    scores = None
    for q in rerank_queries:
        s = cross_encoder.predict([[scoring_query(q), scoring_text(doc.page_content)] for doc in merged_list])
        scores = s if scores is None else [max(a, b) for a, b in zip(scores, s)]
    scored_docs = sorted(zip(scores, merged_list), key=lambda x: x[0], reverse=True)

    #The cross-encoder only reads the first 512 tokens of a chunk, so it can bury a long chunk whose
    #answer is near the end (e.g. the employee count). Always keep the top BM25 and semantic hits.
    must_keep = {doc.page_content for doc in bm25_docs[:KEEP_PER_SEARCH] + sem_docs[:KEEP_PER_SEARCH]}
    kept = [pair for pair in scored_docs if pair[1].page_content in must_keep]
    rest = [pair for pair in scored_docs if pair[1].page_content not in must_keep]
    final = kept + rest[:max(TOP_K - len(kept), 0)]
    return sorted(final, key=lambda x: x[0], reverse=True)


def retrieve(state: RetrieveAgent) -> RetrieveAgent:
    question = state["question"]
    rewritten = state.get("search_query")
    if rewritten:
        #Second pass: search with the rewrite, but compete against the first pass's chunks
        #so a bad rewrite can only add chunks, never throw away good ones
        scored_docs = hybrid_search(rewritten, [question, rewritten], state.get("documents", []))
    else:
        scored_docs = hybrid_search(question, [question])

    print(f"[retriever] query={rewritten or question!r} top scores={[round(float(s), 2) for s, _ in scored_docs[:3]]}")

    top_score = float(scored_docs[0][0]) if scored_docs else float("-inf")
    return {**state, "documents": [doc for _, doc in scored_docs], "top_score": top_score}


def rewrite_query(state: RetrieveAgent) -> RetrieveAgent:
    prompt = f"""Rewrite this question as a search query over chunks of Apple's fiscal 2024 Form 10-K.
    Expand abbreviations (e.g. R&D → research and development) and use the terms a 10-K would use
    (e.g. "net sales" instead of "revenue" or "made"; "outsourcing partners" for manufacturers).
    Don't include the words "Apple", "10-K", "Form", "fiscal 2024", or section names: every chunk
    contains them, so they don't help find the right one.

    Question: {state["question"]}

    Respond with the rewritten query only."""
    rewritten = llm.invoke(prompt).content.strip().strip('"')
    return {**state, "search_query": rewritten, "rewrite_count": state.get("rewrite_count", 0) + 1}


def should_rewrite(state: RetrieveAgent) -> str:
    if state.get("top_score", float("-inf")) < RELEVANCE_THRESHOLD and state.get("rewrite_count", 0) < MAX_REWRITES:
        return "rewrite"
    return "done"


workflow = StateGraph(RetrieveAgent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite_query", rewrite_query)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_rewrite, {"rewrite": "rewrite_query", "done": END})
workflow.add_edge("rewrite_query", "retrieve")
retriever_app = workflow.compile()
