import argparse
from pathlib import Path

from src.rag_service.tracing import setup_tracing
from src.rag_service.pipelines.ingest import ingest_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="RFP 문서(pdf/hwp)가 있는 디렉토리 경로",
    )
    args = parser.parse_args()

    setup_tracing()
    ingest_documents(Path(args.source_dir))


if __name__ == "__main__":
    main()
