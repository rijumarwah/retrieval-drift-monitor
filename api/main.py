from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline.embedder import Embedder
from rag_pipeline.indexer import get_weaviate_client
from rag_pipeline.query import QueryEngine


# ====== Request/Response models ======
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    score: float

class QueryResponse(BaseModel):
    question: str
    results: list[ChunkResult]


# ====== App setup ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup, loads the embedding model and connects
    to Weaviate. Stored on app.state so the endpoint can access them
    without re-initialising on every request.

    Instead of Global variables which initialize at import time, which breaks tests
    that mock these dependencies, we instead use Lifespan which initializes at startup,
    which tests can skip entirely
    """
    app.state.embedder = Embedder()
    app.state.client = get_weaviate_client()
    app.state.query_engine = QueryEngine(
        embedder=app.state.embedder,
        client=app.state.client,
    )
    yield
    # cleanup on shutdown


app = FastAPI(
    title="Retrieval Drift Monitor",
    description="Query endpoint for the RAG pipeline being monitored.",
    version="0.1.0",
    lifespan=lifespan,
)


# ====== Endpoints ===================

@app.get("/health")
def health_check():
    """
    The monitoring layer calls this before
    running shadow evaluations to confirm the pipeline is up.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Accepts a natural language question, returns top-k chunks.
    Only endpoint the monitoring layer ever calls.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    results = app.state.query_engine.query(
        question=request.question,
        top_k=request.top_k,
    )

    return QueryResponse(question=request.question, results=results)
