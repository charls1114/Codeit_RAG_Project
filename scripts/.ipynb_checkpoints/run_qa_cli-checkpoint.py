# scripts/run_qa_cli.py
from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.qa_chain import build_rag_chain


def main():
    setup_tracing()
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
