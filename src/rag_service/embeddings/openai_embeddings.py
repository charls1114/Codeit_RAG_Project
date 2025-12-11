from langchain_openai import OpenAIEmbeddings
from ..config import get_app_config


def get_openai_embeddings():
    cfg = get_app_config()
    return OpenAIEmbeddings(
        model=cfg.openai_embedding_model,
        api_key=cfg.openai_api_key,
    )
