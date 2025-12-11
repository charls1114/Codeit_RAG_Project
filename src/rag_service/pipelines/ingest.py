from pathlib import Path
from ..loaders import get_document_loader
from ..chunking.splitter import split_documents
from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import create_chroma_from_documents


def ingest_documents(source_dir: str | Path):
    loader = get_document_loader()
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
