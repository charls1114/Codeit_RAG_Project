from pathlib import Path
from ..loaders.multimodal_loader import MultiModalLoader
from ..chunking.splitter import split_documents
from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import create_chroma_from_documents


def ingest_documents(source_dir: str | Path):
    """
    주어진 디렉토리에서 문서를 로드하고, 청크로 분할한 후 Chroma 벡터스토어에 저장합니다.
    Args:
        source_dir: 문서가 저장된 디렉토리 경로
    Returns:
        vectordb: 생성된 Chroma 벡터스토어 객체
    """
    loader = MultiModalLoader()
    print(f"[INGEST] Loading documents from {source_dir} ...")
    docs = loader.load_directory(source_dir)

    print(f"[INGEST] Loaded {len(docs)} documents. Splitting...")
    chunks = split_documents(docs)
    print(f"[INGEST] Created {len(chunks)} chunks.")

    embeddings = get_embeddings()
    print("[INGEST] Creating Chroma vectorstore...")
    vectordb = create_chroma_from_documents(chunks, embeddings)

    print("[INGEST] Done. Vectorstore persisted.")
    return vectordb
