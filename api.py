import logging
from pathlib import Path
from typing import List, Literal, Optional
import openai
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field, field_validator
from agent10 import orchestrator_app, initial_state
from retriever_agent import vs
from cache import ResponseCache, CACHEABLE_ROUTES

logger = logging.getLogger("research_assistant")
logging.basicConfig(level=logging.INFO)

MAX_QUESTION_CHARS = 1000
MAX_HISTORY_MESSAGES = 6  #last 3 exchanges are plenty to resolve follow-ups

app = FastAPI()
cache = ResponseCache()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    chat_history: List[Turn] = Field(default_factory=list, max_length=50)
    company: Optional[str] = None  #sent by the UI, unused until multi-company support

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v


def error_response(status: int, code: str, message: str) -> JSONResponse:
    #Same shape as a normal answer, so the UI can always show data.answer
    return JSONResponse(status_code=status, content={"error": code, "answer": message, "sources": []})


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError):
    return error_response(422, "invalid_request",
                          f"Please enter a question between 1 and {MAX_QUESTION_CHARS} characters.")


@app.post("/ask")
def ask(request: QuestionRequest):
    history = [t.model_dump() for t in request.chat_history][-MAX_HISTORY_MESSAGES:]
    data_version = vs.count()

    #Follow-ups depend on the conversation, so only first questions are cached
    if not history:
        cached = cache.get(request.question, data_version)
        if cached:
            return {**cached, "cached": True}

    try:
        result = orchestrator_app.invoke(initial_state(request.question, history))
    except openai.RateLimitError as e:
        logger.exception("OpenAI rate limit")
        if getattr(e, "code", None) == "insufficient_quota":
            return error_response(503, "quota_exceeded", "The server's OpenAI quota is used up. Contact the administrator.")
        return error_response(429, "rate_limited", "Too many requests right now. Please wait a moment and try again.")
    except (openai.APITimeoutError, openai.APIConnectionError):
        logger.exception("OpenAI unreachable")
        return error_response(503, "model_unavailable", "The language model is unreachable right now. Please try again shortly.")
    except openai.AuthenticationError:
        logger.exception("OpenAI authentication failed")
        return error_response(500, "server_misconfigured", "The server's OpenAI API key is invalid. Contact the administrator.")
    except openai.APIStatusError:
        logger.exception("OpenAI error")
        return error_response(502, "model_error", "The language model returned an error. Please try again.")
    except GraphRecursionError:
        logger.exception("Agent loop limit hit")
        return error_response(500, "agent_loop", "The assistant got stuck answering this question. Try rephrasing it.")
    except Exception:
        logger.exception("Unhandled error answering %r", request.question)
        return error_response(500, "internal_error", "Something went wrong answering your question. Please try again.")

    sources = []
    for doc in result["documents"]:
        sources.append({"content": doc.page_content, "page": doc.metadata.get("page", "")})

    response = {
        "answer": result["answer"],
        "sources": sources,
        "route": result["is_retrieval"],
        "grounded": result.get("grounded", True),
    }
    #Don't cache answers that failed the hallucination check, or we'd keep serving them
    if not history and response["route"] in CACHEABLE_ROUTES and response["grounded"]:
        cache.set(request.question, data_version, response)
    return {**response, "cached": False}
