from langchain_huggingface import HuggingFaceEmbeddings
from ..config import get_app_config


def get_local_hf_embeddings():
    cfg = get_app_config()
    return HuggingFaceEmbeddings(
        model_name=cfg.hf_embedding_model,
        model_kwargs={"device": cfg.device},
        encode_kwargs={"normalize_embeddings": True},
    )
