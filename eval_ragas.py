import time
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from retriever_agent import retriever_app
from writer_agent import writer_app
from ragas.llms import LangchainLLMWrapper                                                                                                                                                
from ragas.embeddings import LangchainEmbeddingsWrapper                                                                                                                                   
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  



ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))                                                                                                                               
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


questions = [
    # Financials
    "What were Apple's total net sales in fiscal year 2024?",
    "What was Apple's net income in fiscal year 2024?",
    "What were Apple's earnings per share in 2024?",
    "How much did Apple spend on research and development in 2024?",
    "What was Apple's provision for income taxes in 2024?",
    "What was Apple's total term debt principal as of September 28, 2024?",
    "What was Apple's gross margin percentage in fiscal year 2024?",
    "What were Apple's total operating expenses in fiscal year 2024?",
    "What was Apple's cash and cash equivalents balance as of September 28, 2024?",
    "How much did Apple return to shareholders through buybacks and dividends in 2024?",
    "What was Apple's effective tax rate in fiscal year 2024?",
    "What were Apple's iPhone net sales in fiscal year 2024?",
    "What were Apple's Services net sales in fiscal year 2024?",
    "What were Apple's Mac net sales in fiscal year 2024?",
    "What were Apple's Wearables, Home and Accessories net sales in fiscal year 2024?",
    "What was Apple's iPad net sales in fiscal year 2024?",
    "What was Apple's capital expenditure in fiscal year 2024?",
    "What was Apple's free cash flow in fiscal year 2024?",
    "What was Apple's selling, general and administrative expense in 2024?",
    "How much commercial paper did Apple have outstanding as of September 28, 2024?",
    # Operations and strategy
    "How many full-time equivalent employees did Apple have as of September 2024?",
    "Who are Apple's largest contract manufacturers?",
    "What products did Apple introduce or update in fiscal year 2024?",
    "How does Apple distribute its products globally?",
    "What is Apple's approach to environmental sustainability as disclosed in the 2024 10-K?",
    "What is Apple's share repurchase program and how much was authorized?",
    "How does Apple generate revenue from its Services segment?",
    "What was the geographic breakdown of Apple's net sales in 2024?",
    "What were Apple's net sales in Greater China in fiscal year 2024?",
    "What were Apple's net sales in Europe in fiscal year 2024?",
    # Risk factors
    "What are the primary risk factors Apple disclosed in its 2024 10-K?",
    "What risks does Apple face related to its supply chain concentration?",
    "What regulatory risks did Apple disclose in its 2024 10-K?",
    "How does Apple manage foreign exchange risk?",
    "What cybersecurity risks did Apple disclose in its 2024 10-K?",
    "What risks does Apple face from competition in the smartphone market?",
    "What risks does Apple face related to its dependence on third-party developers?",
    "What legal proceedings did Apple disclose in its 2024 10-K?",
    "What risks does Apple face from climate change as disclosed in the 10-K?",
    "What risks does Apple face related to intellectual property?",
    # Market risk
    "What was the maximum one-day loss in fair value estimated by Apple's VAR model in 2024?",
    "How does Apple hedge its foreign currency denominated term debt?",
    "What is Apple's exposure to interest rate risk?",
    "How does Apple classify its marketable securities?",
    "What was the fair value of Apple's Notes as of September 28, 2024?",
    # Governance and compensation
    "What stock-based compensation did Apple disclose in its 2024 10-K?",
    "Who are Apple's principal executive officers as listed in the 2024 10-K?",
    "What are the terms of Apple's 2022 Employee Stock Plan?",
    "How does Apple's audit committee oversee financial reporting?",
    "What were the audit fees paid by Apple in fiscal year 2024?",
    "What dividends did Apple pay per share in fiscal year 2024?",
]

all_questions = []
all_contexts = []
all_answers = []

for q in questions:
    retrieve_result = retriever_app.invoke({
        "question": q,
        "documents": [] #docs will get stored when it is invoked, itll go to vector db for question then run bm25+semantic -> crossencosider then return us 4 docs
    })
    
    print(f"Documents Returned: {retrieve_result['documents']}")
    documents = retrieve_result['documents']
    contexts = []
    for i, doc in enumerate(documents):
        print(f"Context {i+1}\t: {doc.page_content}")    
        contexts.append(doc.page_content)

    writer_result = writer_app.invoke({
        "question": q,
        "documents": documents,
        "is_retrieval": "retrieve",
        "chat_history": [],
        "tool_result": "",
        "answer": "",
        "hallucinate_count": 0,
        "rewritten":""
    })
    
    answer = writer_result["answer"]
    print("answer", answer)
    
    
    all_questions.append(q)
    all_contexts.append(contexts)
    time.sleep(5)
    all_answers.append(answer)

dataset = Dataset.from_dict({
    "question": all_questions,
    "contexts": all_contexts,
    "answer": all_answers
})

print("-----------RESULTS START HERE-----------\n\n")
results = evaluate(dataset, metrics = [faithfulness, answer_relevancy], llm = ragas_llm, embeddings = ragas_embeddings)
print(results)                                                                                                                                                                            
print(results.to_pandas())

FAITHFULNESS_THRESHOLD = 0.9
RELEVANCY_THRESHOLD = 0.85

df = results.to_pandas()
faithfulness_score = df["faithfulness"].mean()
relevancy_score = df["answer_relevancy"].mean()

if faithfulness_score < FAITHFULNESS_THRESHOLD:
    raise Exception(f"Regression: faithfulness {faithfulness_score:.4f} < {FAITHFULNESS_THRESHOLD}")
if relevancy_score < RELEVANCY_THRESHOLD:
    raise Exception(f"Regression: answer_relevancy {relevancy_score:.4f} < {RELEVANCY_THRESHOLD}")                                                                                                                                                                                                                                       

print("All regression checks passed.")     