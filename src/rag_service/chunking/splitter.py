from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List


def get_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = get_text_splitter()
    return splitter.split_documents(docs)
