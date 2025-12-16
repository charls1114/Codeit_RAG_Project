import os

from langchain_ollama import OllamaEmbeddings


def get_ollama_embeddings():
    return OllamaEmbeddings(
        model=os.getenv('OLLAMA_EMBEDDING_MODEL'),
        base_url=os.getenv('OLLAMA_URL'),
        num_gpu=0
    )
