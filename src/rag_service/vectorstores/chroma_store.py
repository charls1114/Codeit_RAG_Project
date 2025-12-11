from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from ..config import get_app_config
from pathlib import Path
from typing import List
from langchain_core.documents import Document


def create_chroma_from_documents(
    docs: List[Document],
    embeddings: Embeddings,
    collection_name: str = "rfp_rag",
) -> Chroma:
    cfg = get_app_config()
    persist_dir = Path(cfg.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    vectordb.persist()
    return vectordb


def load_chroma(
    embeddings: Embeddings,
    collection_name: str = "rfp_rag",
) -> Chroma:
    cfg = get_app_config()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.chroma_persist_dir,
    )
