from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents
from src.rag_service.pipelines.qa_chain import build_rag_chain
import sys
from pathlib import Path


def main():
    # 프로젝트 루트 디렉터리를 sys.path에 추가
    ROOT_DIR = Path(__file__).resolve().parents[1]  # Codeit_RAG_Project
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    data_dir = ROOT_DIR / "data" / "raw"
    setup_tracing()
    ingest_documents(data_dir)
    chain = build_rag_chain(k=5)

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
