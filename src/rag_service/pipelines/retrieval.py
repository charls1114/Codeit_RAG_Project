# 파일: src/rag_service/pipelines/retrieval.py
from __future__ import annotations

import re  # [추가] 태그 제거를 위한 정규표현식 모듈
from typing import List, Dict, Tuple, Union, Any
from langchain_core.documents import Document

from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import load_chroma
from ..config import get_app_config


def _dedup_docs(docs: List[Document]) -> List[Document]:
    """
    중복 문서를 제거합니다.
    Args:
        docs: 검색한 Document의 목록
    Returns:
        out: 중복이 제거된 Document의 목록
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
    Args:
        k: 검색할 document의 수
        doc_type: 검색할 document의 type
    Returns:
        Chroma의 retriever 객체
    """
    embeddings = get_embeddings()
    vectordb = load_chroma(embeddings)
    search_kwargs = {"k": k}
    if doc_type:
        search_kwargs["filter"] = {"type": doc_type}

    return vectordb.as_retriever(search_kwargs=search_kwargs)


def retrieve_multi(
    question: Union[str, Dict[str, Any]],  # [수정] str 또는 dict 모두 허용
    k_text: int = 4,
    k_table: int = 3,
    k_image: int = 3,
) -> List[Document]:
    """
    text/table/image를 각각 따로 검색 후 합친 뒤 중복을 제거한 Document의 목록을 반환합니다.
    입력이 딕셔너리일 경우 'input' 키의 텍스트를 사용하여 검색합니다.

    Args:
        question: 검색할 질문 (문자열 또는 딕셔너리)
        k_text: 검색할 텍스트 Document의 수
        k_table: 검색할 테이블 Document의 수
        k_image: 검색할 이미지 Document의 수
    Returns:
        중복이 제거된 Document의 목록
    """

    # [수정 시작] 딕셔너리 입력 처리 및 태그 정제 로직
    if isinstance(question, dict):
        # 1. 딕셔너리에서 실제 질문 내용 추출
        query_text = question.get("input", "")

        # 2. 검색 정확도를 위해 태그(<USER_QUESTION> 등) 제거
        # 예: "<USER_QUESTION> 안녕 </USER_QUESTION>" -> "안녕"
        if isinstance(query_text, str):
            query_text = re.sub(r"<[^>]+>", "", query_text).strip()
    else:
        # 문자열인 경우 그대로 사용
        query_text = question
    # [수정 끝]

    docs: List[Document] = []

    # 텍스트 / 테이블 / 이미지 각각 따로 검색 (question 대신 query_text 사용)
    if k_text > 0:
        docs.extend(get_retriever(k=k_text, doc_type="text").invoke(query_text))
    if k_table > 0:
        docs.extend(get_retriever(k=k_table, doc_type="table").invoke(query_text))
    if k_image > 0:
        docs.extend(get_retriever(k=k_image, doc_type="image").invoke(query_text))

    return _dedup_docs(docs)
