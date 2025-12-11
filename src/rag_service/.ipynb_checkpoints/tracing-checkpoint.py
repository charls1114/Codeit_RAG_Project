# src/rag_service/tracing.py
import os
from .config import get_app_config

def setup_tracing():
    cfg = get_app_config()
    if cfg.langsmith_tracing and cfg.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langsmith_api_key
        if cfg.langsmith_project:
            os.environ["LANGCHAIN_PROJECT"] = cfg.langsmith_project
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
