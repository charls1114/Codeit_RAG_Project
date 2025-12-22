from pathlib import Path
from typing import List

from langchain_core.documents import Document
from .base import BaseRFPDocumentLoader

from llama_index.readers.file import HWPReader, PyMuPDFReader


class LlamaIndexRFPDocumentLoader(BaseRFPDocumentLoader):
    def __init__(self):
        self.hwp_reader = HWPReader()
        self.pdf_reader = PyMuPDFReader()

    def _convert_to_langchain_docs(self, li_docs) -> List[Document]:
        lc_docs: List[Document] = []
        for d in li_docs:
            lc_docs.append(
                Document(
                    page_content=d.text_resource.text,
                    metadata={**(d.metadata or {}), "doc_type": "rfp"},
                )
            )
        return lc_docs

    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        # llama_index의 load_data는 페이지 구분 없이 문서 전체를 한번에 로드함
        if path.suffix.lower() == ".hwp":
            li_docs = self.hwp_reader.load_data(file=path)
        elif path.suffix.lower() == ".pdf":
            li_docs = self.pdf_reader.load_data(file_path=path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        return self._convert_to_langchain_docs(li_docs)

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        dir_path = Path(dir_path)
        all_docs = []
        for fp in dir_path.glob("**/*"):
            all_docs.extend(self.load(fp))
        return all_docs
