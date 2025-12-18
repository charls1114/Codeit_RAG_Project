from __future__ import annotations

from typing import List, Dict, Tuple
from langchain_core.documents import Document

from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import load_chroma


def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen = set()
    out = []
    for d in docs:
        m = d.metadata or {}
        key = (
            m.get("source"),
            m.get("page"),
            m.get("type"),
            m.get("image_path"),
            m.get("table_index"),
            (d.page_content[:200] if d.page_content else ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def get_retriever(k: int = 5, doc_type: str | None = None):
    embeddings = get_embeddings()
    vectordb = load_chroma(embeddings)
    search_kwargs = {"k": k}
    if doc_type:
        search_kwargs["filter"] = {"type": doc_type}

    return vectordb.as_retriever(search_kwargs=search_kwargs)


def retrieve_multi(
    question: str, k_text: int = 4, k_table: int = 3, k_image: int = 3
) -> List[Document]:
    """
    - text/table/image 각각 따로 검색
    - 합친 뒤 중복제거
    """
    docs: List[Document] = []
    if k_text > 0:
        docs.extend(get_retriever(k=k_text, doc_type="text").invoke(question))
    if k_table > 0:
        docs.extend(get_retriever(k=k_table, doc_type="table").invoke(question))
    if k_image > 0:
        docs.extend(get_retriever(k=k_image, doc_type="image").invoke(question))

    return _dedup_docs(docs)
