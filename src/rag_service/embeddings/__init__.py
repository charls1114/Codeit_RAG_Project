from ..config import get_app_config
from .local_hf_embeddings import get_local_hf_embeddings
from .openai_embeddings import get_openai_embeddings


def get_embeddings():
    cfg = get_app_config()
    if cfg.rag_mode == "local_hf":
        return get_local_hf_embeddings()
    elif cfg.rag_mode == "openai_api":
        return get_openai_embeddings()
    else:
        raise ValueError(f"지원하지 않는 RAG_MODE: {cfg.rag_mode}")
