from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from pathlib import Path


class BaseRFPDocumentLoader(ABC):
    @abstractmethod
    def load(self, path: str | Path) -> List[Document]:
        """
        단일 파일을 LangChain Document 리스트로 로드합니다.
        Args:
            path: 단일 파일 경로
        Returns:
            List[Document]: 로드된 Document 객체들의 리스트
        """
        raise NotImplementedError

    @abstractmethod
    def load_directory(self, dir_path: str | Path) -> List[Document]:
        """
        디렉토리 내부의 모든 RFP 파일(pdf/hwp)을 로드합니다.
        Args:
            dir_path: 디렉토리 경로
        Returns:
            List[Document]: 디렉토리 내 모든 파일이 로드된 Document 객체들의 리스트
        """
        raise NotImplementedError
