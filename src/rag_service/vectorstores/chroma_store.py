from langchain_chroma.vectorstores import Chroma
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
    """
    주어진 문서들로 Chroma 벡터저장소를 생성하고 저장합니다.
    Args:
        docs: Document 목록
        embeddings: 임베딩 모델
        collection_name: 컬렉션 이름
    Returns:
        Chroma 벡터저장소
    """
    cfg = get_app_config()
    # 벡터저장소 디렉토리 없을시 생성
    persist_dir = Path(cfg.vectorstore.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    # 입력받은 문서로 벡터저장소 생성
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    return vectordb


def load_chroma(
    embeddings: Embeddings,
    collection_name: str = "rfp_rag",
) -> Chroma:
    """
    폴더에 저장된 Chroma 벡터저장소를 로드합니다.
    Args:
        embeddings: 임베딩 모델
        collection_name: 컬렉션 이름
    Returns:
        Chroma 벡터저장소
    """
    cfg = get_app_config()
    # 폴더에 저장된 벡터저장소 로드
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=cfg.vectorstore.persist_dir,
    )
