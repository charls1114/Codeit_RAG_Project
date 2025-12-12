from pathlib import Path
from typing import List

from langchain_core.documents import Document
from .base import BaseRFPDocumentLoader

# 예시용: 실제 사용 시 공식 문서 보고 import 수정
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_teddynote.document_loaders import HWPLoader  # 예: 실제 패키지 이름 확인 필요
from helper_hwp import hwp_to_txt


class PyMuPDFTeddynoteRFPDocumentLoader(BaseRFPDocumentLoader):
    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
        elif path.suffix.lower() == ".hwp":
            loader = HWPLoader(str(path))
            docs = loader.load()
            # txts = hwp_to_txt(str(path))
            # docs = [Document(page_content=txt) for txt in txts]
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
