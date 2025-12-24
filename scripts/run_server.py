# 파일: scripts/run_server.py
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# --- [모듈 임포트] ---
from src.rag_service.models.schemas import QuestionRequest
from src.rag_service.core.startup import initialize_vector_db, get_retriever_and_chain
from src.rag_service.pipelines.qa_chain import build_rag_chain
from src.rag_service.services.chat_flow import ChatService

# =================================================================
# 📍 전역 변수
# =================================================================
chat_service = None  # 모든 로직을 담고 있는 매니저 객체

# 경로 설정
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
static_dir = project_root / "static"
data_dir = Path("/home/public/data")
raw_data_path = data_dir / "raw_data"
# =================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # [1] 전역 변수 호출
    # 함수 밖에서도 이 변수(chat_service)를 계속 써야 하므로 'global'로 선언합니다.
    # 만약 global을 안 쓰면, 이 함수가 끝날 때 chat_service 변수도 같이 사라져 버립니다.
    global chat_service

    # [2] 데이터베이스(DB) 안전 점검
    # startup.py에 있는 함수를 호출합니다.
    # 실제 역할: "DB 폴더가 비었나? 비었으면 raw_data 폴더에서 문서를 읽어서 채워넣어라."
    # 서버가 켜지기 전에 데이터가 준비되어 있어야 하므로 가장 먼저 실행합니다.
    initialize_vector_db(raw_data_path)

    # [3] 핵심 부품 조달 (Factory Pattern)
    # startup.py의 함수를 호출하여 두 가지 핵심 도구를 받아옵니다.
    # - retriever: "도서관 사서" (문서 찾는 도구)
    # - chain: "AI 작가" (답변 쓰는 도구)
    # 이 과정에서 OpenAI API 연결, ChromaDB 연결 등이 내부적으로 일어납니다.
    retriever, chain = get_retriever_and_chain(build_rag_chain)

    # [4] 서비스 매니저 조립 (Dependency Injection)
    # 여기가 제일 중요합니다.
    # ChatService라는 "총괄 매니저"를 고용하는데, 빈손으로 고용하는 게 아닙니다.
    # 위에서 구한 도구(retriever, chain)를 손에 쥐여주면서 생성합니다.
    # 이제 ChatService는 이 도구들을 가지고 평생(서버 켜져있는 동안) 일합니다.
    chat_service = ChatService(retriever=retriever, chain=chain)

    print("🚀 서버 준비 완료! ChatService 가동 중...")

    # [5] 일시 정지 (Yield)
    # yield는 "양보하다"라는 뜻입니다.
    # 여기서 lifespan 함수의 실행은 '일시 정지' 상태가 되고, 서버의 제어권이 FastAPI로 넘어갑니다.
    # 즉, 이 시점부터 서버는 "영업 시작(Listening)" 상태가 됩니다.
    yield

    # [6] 영업 종료 (Cleanup)
    # 사용자가 서버를 강제로 끄면(Ctrl+C), yield 이후의 코드가 실행됩니다.
    # DB 연결을 끊거나, 로그를 저장하는 등의 마무리 작업을 여기서 합니다.
    print("서버 종료.")


# [1] 앱 생성 및 수명주기 연결
# FastAPI 앱을 만드는데, "이 앱의 시작과 끝은 lifespan 함수가 관리한다"라고 지정해줍니다.
app = FastAPI(lifespan=lifespan)

# [2] 정적 파일 연결 (Mounting)
# "/static"이라는 주소로 들어오는 요청은 static_dir 폴더의 파일을 그대로 보여주라는 뜻입니다.
# 예: 브라우저가 http://.../static/style.css 를 요청하면 -> static 폴더의 style.css를 줌.
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    # [1] index.html 파일 찾기
    index_file = static_dir / "index.html"

    # [2] 파일 존재 여부 방어 코드
    # 만약 index.html이 없으면 404 에러를 띄웁니다.
    if not index_file.exists():
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

    # [3] 파일 읽어서 돌려주기
    # 파일을 열어서(open) 그 안의 HTML 텍스트를 읽은 뒤(read),
    # 브라우저에게 그대로 던져줍니다(return). 브라우저는 이 HTML을 해석해서 화면을 그립니다.
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()


# [1] 요청 모델 정의 (Pydantic)
# request: QuestionRequest -> "들어오는 데이터는 무조건 QuestionRequest(schemas.py) 모양이어야 해"
# 만약 사용자가 이상한 데이터를 보내면 FastAPI가 알아서 에러를 뱉습니다.
@app.post("/api/chat")
async def chat_endpoint(request: QuestionRequest):
    """
    Controller Layer (컨트롤러 계층)
    - 역할: 요청을 받고(Input), 일꾼에게 시키고(Process), 결과를 돌려줌(Output).
    - 절대로 여기서 복잡한 계산을 하지 않습니다.
    """
    # 전역 변수로 만들어둔 서비스 매니저를 불러옵니다.
    global chat_service

    # [2] 업무 위임 (Delegation)
    # "야, 서비스 매니저(chat_service)! 손님이 질문(request.query) 가져왔어. 답변 좀 만들어봐."
    # 모든 지지고 볶는 과정(검색, 메모리, 생성)은 generate_reply 함수 안에서 일어납니다.
    # 서버 코드는 그 과정에 대해 알 필요가 없습니다. (캡슐화)
    answer = chat_service.generate_reply(request.query)

    # [3] 결과 반환
    # 서비스 매니저가 준 답변을 JSON 형태로 포장해서 손님에게 건네줍니다.
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
