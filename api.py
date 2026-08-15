from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent10 import orchestrator_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse("index.html")


class QuestionRequest(BaseModel):
    question: str
    

@app.post("/ask")
def ask(request: QuestionRequest):
    result = orchestrator_app.invoke({
        "question": request.question,
        "documents": [],
        "tool_result": "",
        "is_retrieval": "",
        "chat_history": [],
        "answer": "",
        "hallucinate_count": 0
    })
    sources = []
    for doc in result["documents"]:
        sources.append({"content": doc.page_content, "page": doc.metadata.get("page", "")})
    
    answer = result["answer"]
    return {"answer": answer, "sources": sources}