import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# 프로젝트 루트 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# FastAPI 관련 임포트
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# --- 기존 모듈 임포트 ---
from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.pipelines.qa_chain import build_rag_chain
from src.rag_service.config import get_app_config

# [추가] DB 직접 검색을 위한 라이브러리
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 전역 변수
rag_chain = None
retriever_for_check = None  # [추가] 데이터가 있는지 미리 찔러볼 검색기
chat_history = []  # [추가] 대화 내용을 저장할 리스트 (휘발성)

# =================================================================
# 📍 경로 설정
# =================================================================
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
static_dir = project_root / "static"

# 권한 문제로 수정 불가능한 데이터 경로는 그대로 둡니다.
data_dir = Path("/home/public/data")
raw_data_path = data_dir / "raw_data"
# =================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, retriever_for_check
    print("서버 시작 중... 설정 및 모델 로딩...")

    cfg = get_app_config()
    chroma_db_path = cfg.vectorstore.persist_dir
    setup_tracing()

    # 1. 문서 적재 시도 (권한 없으면 실패할 수 있으니 try-except 감싸기)
    if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
        print("⚠️ DB가 비어있어 보입니다. 적재를 시도합니다.")
        try:
            if raw_data_path.exists():
                ingest_documents(raw_data_path)
            else:
                print(f"❌ 데이터 폴더 없음: {raw_data_path}")
        except Exception as e:
            print(f"⚠️ 문서 적재 중 에러 발생 (권한 문제 등): {e}")
            print("👉 기존 DB를 읽기 전용으로 사용하거나, 빈 상태로 시작합니다.")

    # 2. [핵심] 검색기(Retriever) 별도 생성
    # RAG 체인과 별개로, '문서가 진짜 있나?' 확인용으로 씁니다.
    try:
        embedding_function = OpenAIEmbeddings(model=cfg.embeddings.model_name)
        vectorstore = Chroma(
            persist_directory=chroma_db_path,
            embedding_function=embedding_function,
            collection_name=cfg.vectorstore.collection_name,
        )
        # 검색기 생성 (유사도 점수 기반)
        retriever_for_check = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.3},  # 정확도 0.3 미만이면 무시
        )
        print("✅ 데이터 확인용 검색기(Retriever) 준비 완료")
    except Exception as e:
        print(f"❌ 검색기 초기화 실패: {e}")

    # 3. 체인 생성
    rag_chain = build_rag_chain(
        k_text=cfg.retrieval.k_text,
        k_table=cfg.retrieval.k_table,
        k_image=cfg.retrieval.k_image,
    )
    print("🚀 서버 준비 완료!")

    yield
    print("서버 종료.")


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


class QuestionRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/chat")
async def chat_endpoint(request: QuestionRequest):
    global rag_chain, retriever_for_check, chat_history

    user_query = request.query

    # ---------------------------------------------------------
    # 1단계: 참고자료 존재 여부 확인 (Pre-check)
    # ---------------------------------------------------------
    if retriever_for_check:
        # DB에서 가장 비슷한 문서 1개를 찾아봅니다.
        docs = retriever_for_check.invoke(user_query)

        # 문서가 하나도 안 잡히면 바로 거절 메시지 리턴
        if not docs:
            print(f"📭 검색 결과 없음: '{user_query}'")
            return {"answer": "참고자료를 찾지 못했습니다."}

    # ---------------------------------------------------------
    # 2단계: 메모리(이전 대화) 적용
    # ---------------------------------------------------------
    # 최근 대화 2턴(질문+답변 2세트)만 요약해서 가져옵니다. (용량 절약)
    recent_history = chat_history[-4:]
    history_text = "\n".join(recent_history)

    # 질문을 [이전 대화 요약 + 현재 질문] 형태로 수정해서 AI에게 던집니다.
    augmented_query = f"""
    [이전 대화 내용 참고]
    {history_text}

    [현재 질문]
    {user_query}
    """

    # (디버깅용) 실제로 AI에게 들어가는 질문 출력
    print(f"📝 입력 프롬프트:\n{augmented_query}")

    # ---------------------------------------------------------
    # 3단계: RAG 답변 생성
    # ---------------------------------------------------------
    if rag_chain is None:
        return {"answer": "모델이 아직 준비되지 않았습니다."}

    # 수정된 질문(augmented_query)을 넣습니다.
    # 만약 AI가 프롬프트를 그대로 읊는다면 request.query를 그대로 쓰되,
    # 문맥 유지가 안 될 수 있습니다. (현재 방식이 가장 호환성이 좋습니다.)
    answer = rag_chain.invoke(augmented_query)
    final_answer = str(answer)

    # ---------------------------------------------------------
    # 4단계: 메모리에 저장 (휘발성)
    # ---------------------------------------------------------
    chat_history.append(f"Q: {user_query}")
    chat_history.append(f"A: {final_answer}")

    # 메모리가 너무 길어지면 앞에서부터 자름 (최대 10개 문장 유지)
    if len(chat_history) > 10:
        chat_history.pop(0)

    return {"answer": final_answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
