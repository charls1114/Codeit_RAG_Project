from ..config import get_app_config
from .local_hf_llm import get_local_hf_llm
from .ollama_llm import get_ollama_llm
from .openai_llm import get_openai_llm


def get_llm():
    cfg = get_app_config()
    if cfg.rag_mode == "local_hf":
        return get_local_hf_llm()
    elif cfg.rag_mode == "openai_api":
        return get_openai_llm()
    elif cfg.rag_mode == "ollama_api":
        return get_ollama_llm()
    else:
        raise ValueError(f"지원하지 않는 RAG_MODE: {cfg.rag_mode}")
