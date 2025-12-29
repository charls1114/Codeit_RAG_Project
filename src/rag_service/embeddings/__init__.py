from ..config import get_app_config
from .local_hf_embeddings import get_local_hf_embeddings
from .openai_embeddings import get_openai_embeddings


def get_embeddings():
    """
    RAG_MODE에 따라 적절한 임베딩 모델을 반환합니다.
    Returns:
        임베딩 모델 인스턴스
    """
    cfg = get_app_config()
    if cfg.rag_mode == "local_hf":
        return get_local_hf_embeddings()
    elif cfg.rag_mode == "openai_api":
        return get_openai_embeddings()
    else:
        raise ValueError(f"지원하지 않는 RAG_MODE: {cfg.rag_mode}")
