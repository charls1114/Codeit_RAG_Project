# 파일 위치: src/rag_service/startup.py
import os
from pathlib import Path

# 절대 경로 임포트 (프로젝트 구조 기준)
from src.rag_service.config import get_app_config
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.tracing import setup_tracing


def initialize_vector_db(raw_data_path: Path):
    """
    서버 시작 시 DB를 확인하고, 비어있으면 데이터를 적재합니다.
    Args:
        raw_data_path: 원본 데이터가 저장된 폴더 경로
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
