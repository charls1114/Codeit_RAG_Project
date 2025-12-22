from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.pipelines.qa_chain import build_rag_chain
from src.rag_service.config import get_app_config
import sys
import os
from pathlib import Path


def main():
    cfg = get_app_config()
    data_dir = Path("/home/public/data")
    raw_data_path = data_dir / "raw_data"
    # 폴더 내 파일 목록 가져오기
    chroma_db_path = cfg.vectorstore.persist_dir
    os.makedirs(chroma_db_path, exist_ok=True)
    items = os.listdir(chroma_db_path)
    # 파일이 존재하는지 확인
    file_exists = False
    for item in items:
        item_path = os.path.join(chroma_db_path, item)
        if os.path.isfile(item_path):  # 해당 항목이 파일이면 True 반환
            file_exists = True
            print(f"폴더 '{chroma_db_path}'에 파일이 있습니다: {item}")
            break  # 파일 하나만 찾아도 멈춤
    setup_tracing()
    if not file_exists:
        ingest_documents(raw_data_path)
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
