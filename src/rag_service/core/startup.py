# 파일 위치: src/rag_service/startup.py
import os
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 절대 경로 임포트 (프로젝트 구조 기준)
from src.rag_service.config import get_app_config
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.tracing import setup_tracing


def initialize_vector_db(raw_data_path: Path):
    """
    서버 시작 시 DB를 확인하고, 비어있으면 데이터를 적재합니다.
    """
    print("🔄 [STARTUP] 초기화 작업 시작: DB 상태 점검 중...")
    setup_tracing()  # 로그 추적 설정

    cfg = get_app_config()  # 설정 파일(config)을 읽어옵니다.
    chroma_db_path = cfg.vectorstore.persist_dir  # DB 저장 경로 확인

    # DB 폴더가 없거나 비어있으면 적재 시도
    if not os.path.exists(chroma_db_path) or not os.listdir(chroma_db_path):
        print("⚠️ [STARTUP] DB가 비어있어 보입니다. 적재를 시도합니다.")
        try:
            # 원본 데이터 폴더가 진짜 있는지 확인
            if raw_data_path.exists():
                ingest_documents(raw_data_path)  # 데이터 적재 실행 (ingest.py 호출)
            else:
                print(f"❌ [STARTUP] 데이터 폴더 없음: {raw_data_path}")
        except Exception as e:
            # 권한 문제 등으로 실패해도 서버가 죽지 않도록 방어
            print(f"⚠️ [STARTUP] 문서 적재 중 에러 발생: {e}")
            print("👉 [STARTUP] 기존 DB를 읽기 전용으로 사용하거나, 빈 상태로 시작합니다.")


def get_retriever_and_chain(build_chain_func):
    """
    서버 시작 시 검색기(Retriever)와 RAG 체인(Chain)을 준비하는 함수입니다.

    Args:
        build_chain_func: qa_chain.py에서 정의한 체인 생성 함수를 인자로 받습니다.
                          (의존성 주입 방식을 사용하여 결합도를 낮춤)

    Returns:
        retriever_for_check: 문서 존재 여부 확인용 검색기 (실패 시 None)
        rag_chain: 답변 생성용 메인 체인
    """

    # =========================================================
    # [1] 설정 및 경로 로드
    # =========================================================
    # 전체 앱의 설정 정보(config)를 가져옵니다. (DB 경로, 모델명 등 포함)
    cfg = get_app_config()

    # ChromaDB가 실제로 저장되어 있는 디스크 상의 폴더 경로를 변수에 담습니다.
    # 예: ./files/chroma_db
    chroma_db_path = cfg.vectorstore.persist_dir

    # =========================================================
    # [2] 검색기(Retriever) 초기화 (Pre-check용)
    # =========================================================
    # 만약 아래 try 블록에서 DB 연결에 실패하더라도,
    # 변수가 정의되지 않아 프로그램이 죽는 것을 방지하기 위해 미리 None으로 초기화합니다.
    retriever_for_check = None

    try:
        print(f"🔍 [STARTUP] 임베딩 모델 로딩 중: {cfg.embeddings.model_name}")
        # 1. 임베딩 모델 준비: 텍스트를 벡터(숫자)로 변환해주는 도구입니다.
        embedding_function = OpenAIEmbeddings(model=cfg.embeddings.model_name)

        # 2. 벡터 저장소(VectorStore) 연결:
        # 디스크에 저장된 ChromaDB 데이터를 불러와서 연결합니다.
        vectorstore = Chroma(
            persist_directory=chroma_db_path,  # 데이터가 저장된 경로
            embedding_function=embedding_function,  # 사용할 임베딩 모델
            collection_name=cfg.vectorstore.collection_name,  # DB 내부 컬렉션 이름
        )

        # 3. 검색기(Retriever) 생성:
        # 단순히 문서를 찾는 게 아니라, '유사도 점수'를 기준으로 필터링합니다.
        # score_threshold=0.3: 유사도가 30% 미만인 문서는 아예 검색 결과에서 제외합니다.
        # k=1: 가장 비슷한 문서 딱 1개만 가져와서 확인합니다. (비용 절약)
        retriever_for_check = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.3},
        )
        print("✅ [STARTUP] 데이터 확인용 검색기(Retriever) 준비 완료")

    except Exception as e:
        # DB 파일이 깨졌거나 권한이 없어서 연결에 실패한 경우
        # 서버를 멈추지 않고, 에러 로그만 출력한 뒤 검색기 없이(None) 진행합니다.
        print(f"❌ [STARTUP] 검색기 초기화 실패 (DB 문제 가능성): {e}")

    # =========================================================
    # [3] 메인 RAG 체인 생성 (답변 생성용)
    # =========================================================
    # 인자로 받은 build_chain_func 함수를 실행하여 실제 AI 로직(Chain)을 만듭니다.
    # config에서 설정한 k값(참고할 문서 개수)을 전달합니다.
    rag_chain = build_chain_func(
        k_text=cfg.retrieval.k_text,  # 텍스트 문서 몇 개 볼래?
        k_table=cfg.retrieval.k_table,  # 표 정보 몇 개 볼래?
        k_image=cfg.retrieval.k_image,  # 이미지 정보 몇 개 볼래?
    )

    # =========================================================
    # [4] 결과 반환
    # =========================================================
    # 완성된 두 가지 도구(검색기, 체인)를 튜플 형태로 반환합니다.
    # 받는 쪽(run_server.py 등)에서는 이 두 가지를 받아서 서비스에 등록합니다.
    return retriever_for_check, rag_chain
