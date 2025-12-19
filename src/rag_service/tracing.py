import os
from .config import get_app_config


def setup_tracing():
    """
    LangSmith 트레이싱 설정을 합니다.
    """
    cfg = get_app_config()
    if cfg.langsmith and cfg.langsmith.api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith.api_key
        if cfg.langsmith.project:
            os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith.project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
