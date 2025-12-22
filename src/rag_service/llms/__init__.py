from ..config import get_app_config
from .local_hf_llm import get_local_hf_llm
from .openai_llm import get_openai_llm


def get_llm():
    """
    RAG_MODE에 따라 적절한 LLM 인스턴스를 가져옵니다.
    """
    cfg = get_app_config()
    if cfg.rag_mode == "local_hf":
        return get_local_hf_llm()
    elif cfg.rag_mode == "openai_api":
        return get_openai_llm()
    else:
        raise ValueError(f"지원하지 않는 RAG_MODE: {cfg.rag_mode}")
