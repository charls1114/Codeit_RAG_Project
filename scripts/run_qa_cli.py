from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.pipelines.qa_chain import build_rag_chain
from src.rag_service.config import get_app_config
import os
from pathlib import Path


def main():
    cfg = get_app_config()
    # 원본 데이터 폴더 경로 지정
    data_dir = Path("/home/public/data")
    raw_data_path = data_dir / "raw_data"
    # 폴더 내 파일 목록 가져오기
    chroma_db_path = Path(cfg.vectorstore.persist_dir)
    os.makedirs(chroma_db_path, exist_ok=True)

    # 벡터 DB 폴더에 DB 파일이 있는지 확인
    file_exists = any(chroma_db_path.iterdir())

    if not file_exists:
        print("벡터 DB가 존재하지 않습니다. 문서 임베딩을 생성합니다...")
        ingest_documents(raw_data_path)
        print("문서 임베딩이 완료되었습니다.")
    else:
        print("벡터 DB가 이미 존재합니다. 임베딩 생성을 건너뜁니다.")

    setup_tracing()
    chain = build_rag_chain(
        k_text=cfg.retrieval.k_text,
        k_table=cfg.retrieval.k_table,
        k_image=cfg.retrieval.k_image,
    )

    print("RFP RAG CLI. 종료하려면 'exit' 입력.")
    while True:
        q = input("\n질문> ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        answer = chain.invoke(q)
        print("\n=== 답변 ===")
        print(answer)


if __name__ == "__main__":
    main()
