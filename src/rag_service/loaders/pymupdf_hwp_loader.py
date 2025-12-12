from pathlib import Path
from typing import List

from langchain_core.documents import Document
from .base import BaseRFPDocumentLoader

from langchain_community.document_loaders import PyMuPDFLoader
from helper_hwp import hwp_to_txt


class PyMuPDFHwpRFPDocumentLoader(BaseRFPDocumentLoader):
    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
        elif path.suffix.lower() == ".hwp":
            txts = hwp_to_txt(str(path))
            docs = [Document(page_content=txts)]
        else:
            raise ValueError(f"지원하지 않는 확장자: {path.suffix}")

        # 공통 메타데이터 추가 (RFP 프로젝트용)
        for d in docs:
            d.metadata.setdefault("source", str(path))
            d.metadata.setdefault("doc_type", "rfp")
        return docs

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        dir_path = Path(dir_path)
        all_docs: List[Document] = []
        for fp in dir_path.glob("**/*"):
            if fp.suffix.lower() in [".pdf", ".hwp"]:
                all_docs.extend(self.load(fp))
        return all_docs
