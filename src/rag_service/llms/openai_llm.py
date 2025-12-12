from langchain_openai import ChatOpenAI
from ..config import get_app_config


def get_openai_llm():
    cfg = get_app_config()
    return ChatOpenAI(
        model=cfg.llm.model_name,
        api_key=cfg.llm.api_key,
        temperature=0.0,
    )
