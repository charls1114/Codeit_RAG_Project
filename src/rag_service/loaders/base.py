from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from pathlib import Path


class BaseRFPDocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str | Path) -> List[Document]:
        """단일 파일을 LangChain Document 리스트로 로드"""
        raise NotImplementedError

    @abstractmethod
    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """디렉토리 내부의 모든 RFP 파일(pdf/hwp)을 로드"""
        raise NotImplementedError
