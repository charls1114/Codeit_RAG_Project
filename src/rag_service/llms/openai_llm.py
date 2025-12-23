from langchain_openai import ChatOpenAI
from ..config import get_app_config


def get_openai_llm():
    """
    OpenAI LLM을 생성합니다.
    Returns:
        ChatOpenAI: OpenAI LLM 객체
    """
    cfg = get_app_config()
    return ChatOpenAI(
        model=cfg.llm.model_name,
        api_key=cfg.model_api_key,
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_new_tokens,
    )
