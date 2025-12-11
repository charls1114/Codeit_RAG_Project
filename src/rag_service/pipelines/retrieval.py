from ..embeddings import get_embeddings
from ..vectorstores.chroma_store import load_chroma


def get_retriever(k: int = 5):
    embeddings = get_embeddings()
    vectordb = load_chroma(embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever
