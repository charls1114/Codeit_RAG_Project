from langchain_huggingface import HuggingFaceEmbeddings
from ..config import get_app_config


def get_local_hf_embeddings():
    cfg = get_app_config()
    return HuggingFaceEmbeddings(
        model_name=cfg.embeddings.model_name,
        model_kwargs={
            "device": cfg.device,
            # jinaai
            "trust_remote_code": True,
            "token": cfg.model_api_key,
        },
        encode_kwargs={"normalize_embeddings": True},
    )
