from pathlib import Path
from typing import List

from langchain_core.documents import Document
from .base import BaseRFPDocumentLoader

# 실제 패키지 이름/클래스는 설치 후 확인 필요 (예시)
from llama_index.readers.file import FlatReader  # or SimpleFileReader / SimpleDirectoryReader


class LlamaIndexRFPDocumentLoader(BaseRFPDocumentLoader):
    def __init__(self):
        # FlatReader는 다양한 파일 타입 지원 (pdf, hwp 등)
        self.reader = FlatReader()

    def _convert_to_langchain_docs(self, li_docs) -> List[Document]:
        lc_docs: List[Document] = []
        for d in li_docs:
            # d.text, d.metadata 가 있다고 가정
            lc_docs.append(
                Document(
                    page_content=d.text,
                    metadata={**(d.metadata or {}), "doc_type": "rfp"},
                )
            )
        return lc_docs

    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        li_docs = self.reader.load_data(file=path)
        return self._convert_to_langchain_docs(li_docs)

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        dir_path = Path(dir_path)
        li_docs = self.reader.load_data(dir_path=dir_path)
        return self._convert_to_langchain_docs(li_docs)
