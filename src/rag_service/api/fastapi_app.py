from fastapi import FastAPI
from pydantic import BaseModel

from ..tracing import setup_tracing
from ..pipelines.qa_chain import build_rag_chain

setup_tracing()
app = FastAPI(title="RFP RAG API")
rag_chain = build_rag_chain(k=5)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    answer = rag_chain.invoke(req.question)
    return QueryResponse(answer=answer)
