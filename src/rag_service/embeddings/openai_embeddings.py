from langchain_openai import OpenAIEmbeddings
from ..config import get_app_config


def get_openai_embeddings():
    """
    OpenAI 임베딩 모델의 인스턴스를 가져옵니다.
    """
    cfg = get_app_config()
    return OpenAIEmbeddings(
        model=cfg.embeddings.model_name,
        api_key=cfg.model_api_key,
    )
