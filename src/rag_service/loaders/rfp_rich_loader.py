from pathlib import Path
from typing import List
from langchain_core.documents import Document

from .base import BaseRFPDocumentLoader
from .pdf_rich_loader import PDFRichLoader, PDFRichLoaderConfig
from .hwp_to_pdf_loader import HWPToPDFThenParseLoader


class RFPRichDocumentLoader(BaseRFPDocumentLoader):
    """
    RFPRichDocumentLoader는 PDF와 HWP 문서를 로드하는 클래스입니다.
    이 클래스는 PDF와 HWP 파일을 로드하고, 그 내용을 Document 형태로 반환합니다.
    HWP 파일은 PDF로 변환 후 로드됩니다.
    """

    def __init__(self, cfg: PDFRichLoaderConfig):
        self.pdf_loader = PDFRichLoader(cfg)
        self.hwp_loader = HWPToPDFThenParseLoader(cfg)

    def load(self, path: str | Path) -> List[Document]:
        path = Path(path)
        suf = path.suffix.lower()

        if suf == ".pdf":
            return self.pdf_loader.load(path)
        elif suf in [".hwp", ".hwpx"]:
            return []
            # return self.hwp_loader.load(path)
        else:
            raise ValueError(f"지원하지 않는 확장자: {suf}")

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        dir_path = Path(dir_path)
        all_docs: List[Document] = []
        for fp in dir_path.glob("**/*"):
            if fp.suffix.lower() in [".pdf", ".hwp", ".hwpx"]:
                all_docs.extend(self.load(fp))
        return all_docs
