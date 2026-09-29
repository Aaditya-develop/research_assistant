from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from vectorstore import VectorStore
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langfuse.langchain import CallbackHandler
from langfuse import get_client, propagate_attributes
from dotenv import load_dotenv

load_dotenv()

langfuse = get_client()

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOpenAI(model = "gpt-4o", temperature = 0)
vs = VectorStore()

class RetrieveAgent(TypedDict): 
    question: str
    documents: List[Document]

def retrieve(state: RetrieveAgent) -> RetrieveAgent:
    question = state["question"]
    
    #BM25 search
    all_docs = vs.get_all_documents() #Contains Documents, metadata
    tokenized_courpus = []
    for doc in all_docs:
        tokenized_courpus.append(doc.page_content.split()) #Split into words

    
    bm25 = BM25Okapi(tokenized_courpus)           #bm25 ranks each word by how often it appears                                                                      
    tokenized_query = question.split()  
    bm25_scores = bm25.get_scores(tokenized_query)   #It takes your tokenized query — the list of words — and compares it against every document in the index you built. Gives a score for every chunk
    bm25_sorted = []
    
    for i in range(len(bm25_scores)):
        bm25_sorted.append((i, bm25_scores[i]))

    bm25_sorted.sort(key=lambda x: x[1], reverse=True)     #Return descending sort of top K items

    bm25_docs = []
    for i, score in bm25_sorted[:8]:
        bm25_docs.append(all_docs[i])
    
    # print("------------BM25 SEARCH BEGINS HERE----------------\n\n")
    # for i, doc in enumerate(bm25_docs):
    #     print(f"Chunk # {i+1}--------Doc content: {doc.page_content[:200]} ")
    # print("------------BM25 SEARCH ENDS HERE----------------\n\n")

    #Semantic Search
    merged_list = []
    sem_search = vs.search_with_scores(question, k = 8)
    
    
    sem_docs = []
    for doc, distance in sem_search:
        sem_docs.append(doc)
    
    # print("------------SEMANTIC SEARCH BEGINS HERE----------------\n\n")
    # for i, doc in enumerate(sem_docs):
    #     print(f"Chunk # {i+1}--------Doc content: {doc.page_content[:200]} ")
    # print("------------SEMANTIC SEARCH ENDS HERE----------------\n\n")

    
    #MERGE BOTH
    seen = set()
    #FIrst loop over sem_docs, add them all too seen.
    for doc in sem_docs:
        seen.add(doc.page_content)
        merged_list.append(doc)
    
    #Now add over the bm25 docs, the ones that are caught be seen will not be added
    for doc in bm25_docs:
        if doc.page_content not in seen:
            merged_list.append(doc)
        # else:
        #     print("caught a similar doc")

    #Reranker: Stores doc.page_content and question, then returns a score
    pairs = []
    for doc in merged_list:
        pairs.append([question, doc.page_content])   #Store the question with the page content we extracted
    scores = cross_encoder.predict(pairs)    #Scores come here with the list of doc contexts
    
    scored_docs = []
    for i in range(len(scores)):
        scored_docs.append((scores[i], merged_list[i]))   

    scored_docs.sort(key=lambda x: x[0], reverse=True)   #Sort top 4 descending same as above
    
    print("------------RERANKER BEGINS HERE----------------\n\n")
    for score, doc in scored_docs:
        print(f"Score: {score}, Doc: {doc.page_content}, Metadata: {doc.metadata}")
    print("------------RERANKER ENDS HERE----------------\n\n")
    
    reranked_docs = []
    for score, doc in scored_docs[:8]:
        reranked_docs.append(doc)
    
    
    
    return{**state, "documents": reranked_docs}


workflow = StateGraph(RetrieveAgent)
workflow.add_node("retrieve", retrieve)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", END)
retriever_app = workflow.compile()