from __future__ import annotations
from pathlib import Path
import subprocess
from typing import List

from langchain_core.documents import Document
from .pdf_rich_loader import PDFRichLoader, PDFRichLoaderConfig


class HWPToPDFThenParseLoader:
    """
    HWP/HWPX를 LibreOffice(headless)로 PDF 변환 후,
    PDFRichLoader로 텍스트/표/이미지까지 추출합니다.
    """

    def __init__(self, pdf_loader_cfg: PDFRichLoaderConfig):
        self.pdf_loader = PDFRichLoader(pdf_loader_cfg)

    def load(self, hwp_path: str | Path) -> List[Document]:
        hwp_path = Path(hwp_path)
        pdf_path = self._convert_to_pdf(hwp_path)
        return self.pdf_loader.load(pdf_path)

    def _convert_to_pdf(self, hwp_path: Path) -> Path:
        out_dir = hwp_path.parent
        print(out_dir)
        # soffice 필요: libreoffice 설치 필요
        cmd = [
            "soffice",
            "--headless",
            "--convert-to",
            "pdf",
            str(hwp_path),
            "--outdir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)

        pdf_path = out_dir / f"{hwp_path.stem}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 변환 결과가 없습니다: {pdf_path}")
        return pdf_path
