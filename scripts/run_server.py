import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# FastAPI 관련 임포트
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- [중요] 기존 run_qa_cli.py에 있던 임포트들을 그대로 가져옵니다 ---
from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.pipelines.qa_chain import build_rag_chain
from src.rag_service.config import get_app_config

# 전역 변수로 chain을 선언해둡니다 (서버가 켜져있는 동안 계속 쓰기 위해)
rag_chain = None


# --- 서버 시작 시 1번만 실행될 '준비 과정' ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    print("서버 시작 중... RAG 모델을 로딩합니다.")

    # 1. 설정 및 경로 로드 (기존 코드와 동일)
    cfg = get_app_config()
    data_dir = Path("/home/public/data")  # 혹은 cfg에서 가져오거나 환경에 맞게
    raw_data_path = data_dir / "raw_data"
    chroma_db_path = cfg.vectorstore.persist_dir

    # 2. DB 폴더/파일 확인 로직 (기존 코드 복사)
    os.makedirs(chroma_db_path, exist_ok=True)
    items = os.listdir(chroma_db_path)
    file_exists = False
    for item in items:
        item_path = os.path.join(chroma_db_path, item)
        if os.path.isfile(item_path):
            file_exists = True
            print(f"폴더 '{chroma_db_path}'에 파일이 있습니다: {item}")
            break

    setup_tracing()

    # 3. 문서가 없으면 적재 (Ingest)
    if not file_exists:
        print("Vector DB가 비어있습니다. 문서 적재를 시작합니다...")
        ingest_documents(raw_data_path)

    # 4. 체인 생성 (이게 핵심 엔진!)
    rag_chain = build_rag_chain(
        k_text=cfg.retrieval.k_text,
        k_table=cfg.retrieval.k_table,
        k_image=cfg.retrieval.k_image,
    )
    print("RAG 모델 로딩 완료! 서버가 준비되었습니다.")

    yield  # 여기서부터 서버가 작동합니다.

    # (서버 종료 시 실행될 코드 - 필요하면 추가)
    print("서버를 종료합니다.")


# FastAPI 앱 생성 (lifespan 적용)
app = FastAPI(lifespan=lifespan)

# 정적 파일(HTML/CSS/JS) 경로 연결
app.mount("/static", StaticFiles(directory="static"), name="static")


class QuestionRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # static/index.html 파일이 있어야 합니다.
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat_endpoint(request: QuestionRequest):
    global rag_chain

    if rag_chain is None:
        return {"answer": "모델이 아직 준비되지 않았습니다."}

    # 기존 코드의 chain.invoke(q)와 동일
    answer = rag_chain.invoke(request.query)

    # invoke 결과가 문자열이면 그대로, dict라면 텍스트만 추출해야 할 수 있음
    # 보통 rag_chain의 반환값 구조에 따라 다름 (여기선 문자열이라 가정)
    return {"answer": str(answer)}


if __name__ == "__main__":
    # 포트 8000번에서 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)

    # http://35.227.109.23:8000
