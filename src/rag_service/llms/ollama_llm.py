import os

from langchain_ollama import OllamaLLM


def get_ollama_llm():
    llm = OllamaLLM(
        base_url=os.getenv("OLLAMA_URL"),
        model=os.getenv("OLLAMA_MODEL"),
        temperature=0,
    )
    return llm
