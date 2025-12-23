from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from ..config import get_app_config


def split_documents(docs: List[Document]) -> List[Document]:
    """
    pdf_rich_loader를 사용하는 경우
    - text: chunking 적용
    - image: (보통 짧으니) 기본적으로 chunking 하지 않음 (원하면 옵션으로 확장 가능)
    - table: Markdown 테이블 구조 깨지므로 chunking 제외
    그 외 loader를 사용하는 경우:
    - text: chunking 적용
    Args:
        docs: Document의 list
    Returns:
        out: chunking이 적용된 Document의 list
    """
    cfg = get_app_config()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )

    out: List[Document] = []
    for d in docs:
        dtype = (d.metadata or {}).get("type", "text")
        if dtype in ["table", "table_error"]:
            out.append(d)
        elif dtype == "image":
            # 이미지 문서는 캡션이 이미 구조화되어 있으므로 그대로 유지
            out.append(d)
        else:
            out.extend(splitter.split_documents([d]))

    return out
