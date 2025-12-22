from __future__ import annotations

from typing import List, Dict, Tuple
from langchain_core.documents import Document

from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import load_chroma
from ..config import get_app_config


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """
    중복 문서를 제거합니다.
    - docs: 검색한 Document의 목록
    - return: 중복이 제거된 Document의 목록
    """
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
    """
    chroma db를 로드하고, retriever를 만듭니다. doc_type이 주어지면, 해당 type의 document만 검색합니다.
    - k: 검색할 document의 수
    - doc_type: 검색할 document의 type
    - return: retriever
    """
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
    text/table/image를 각각 따로 검색 후 합친 뒤 중복을 제거한 Document의 목록을 반환합니다.
    - question: 검색할 질문
    - k_text: 검색할 텍스트 Document의 수
    - k_table: 검색할 테이블 Document의 수
    - k_image: 검색할 이미지 Document의 수
    - return: 중복이 제거된 Document의 목록
    """
    docs: List[Document] = []
    cfg = get_app_config()
    # rich loader 사용시 텍스트 / 테이블 / 이미지 각각 따로 검색
    if cfg.document.loader_backend == "pdf_rich_loader":
        if k_text > 0:
            docs.extend(get_retriever(k=k_text, doc_type="text").invoke(question))
        if k_table > 0:
            docs.extend(get_retriever(k=k_table, doc_type="table").invoke(question))
        if k_image > 0:
            docs.extend(get_retriever(k=k_image, doc_type="image").invoke(question))
    # 그 외 loader 사용시 전체를 검색
    else:
        docs.extend(get_retriever(k=k_text).invoke(question))

    return _dedup_docs(docs)
