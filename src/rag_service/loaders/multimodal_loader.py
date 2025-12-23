from __future__ import annotations
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import re
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

from .base import BaseRFPDocumentLoader
from ..config import get_app_config
from ..image_processing.image_to_docs import ImageToDocs


class MultiModalLoader(BaseRFPDocumentLoader):
    """
    PDF 파일에서 텍스트, 테이블, 이미지를 추출하여 Document로 변환합니다.
    """

    def __init__(self):
        app_cfg = get_app_config()
        self.cfg = app_cfg.loader_config

        # ✅ image_processing 설정 반영
        self.ip = app_cfg.loader_config.image_processing
        Path(self.ip.image_output_dir).mkdir(parents=True, exist_ok=True)

        # 이미지 -> 텍스트 요약 모듈 초기화
        self.image_to_docs = ImageToDocs()

        self.TABLE_BLOCK_RE = re.compile(
            r"""
            (                               # table block start
            ^\|.*\|\s*$\n                 # header row
            ^\|(?:\s*:?-+:?\s*\|)+\s*$\n  # separator row: |---| or |:---:|
            (?:^\|.*\|\s*$\n?)+           # body rows (1+)
            )                               # table block end
            """,
            re.MULTILINE | re.VERBOSE,
        )

    def load(self, pdf_path: str | Path) -> List[Document]:
        pdf_path = Path(pdf_path)
        docs: List[Document] = []

        # ✅ 텍스트 추출
        docs.extend(self._extract_text_docs(pdf_path))

        # ✅ 테이블 추출
        if self.cfg.extract_tables:
            docs.extend(self._extract_table_docs(pdf_path))

        # ✅ 이미지 추출
        if self.ip.extract_images:
            docs.extend(self._extract_image_docs(pdf_path))

        return docs

    def load_directory(self, dir_path: str | Path) -> List[Document]:
        dir_path = Path(dir_path)
        all_docs: List[Document] = []
        for fp in dir_path.glob("**/*"):
            if fp.suffix.lower() in [".pdf"]:
                all_docs.extend(self.load(fp))
        return all_docs

    def _extract_text_docs(self, pdf_path: Path) -> List[Document]:
        """
        PDF 파일에서 fitz로 텍스트를 추출하여 Document로 변환합니다.
        Args:
            pdf_path: PDF 파일 경로
        Returns:
            out: Document의 목록
        """
        out: List[Document] = []
        with fitz.open(pdf_path) as doc:
            end = (
                min(doc.page_count, self.cfg.max_pages)
                if self.cfg.max_pages
                else doc.page_count
            )
            for i in range(end):
                page = doc.load_page(i)
                text = page.get_text("text").strip()
                if not text:
                    continue
                out.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(pdf_path),
                            "page": i + 1,
                            "type": "text",
                        },
                    )
                )
        return out

    def _split_md_tables(self, doc: Document):
        """
        Document에서 마크다운 형식의 테이블을 추출합니다.
        Args:
            doc: Document 객체
        Returns:
            tables: 마크다운 형식으로 추출된 테이블
        """
        content = doc.page_content
        tables: list[str] = self.TABLE_BLOCK_RE.findall(content)  # list[str]

        return [t.strip() for t in tables]

    def _extract_table_docs(self, pdf_path: Path) -> List[Document]:
        """
        PDF 파일에서 PyMuPDFLoader로 테이블을 추출하여 Document로 변환합니다.
        Args:
            pdf_path: PDF 파일 경로
        Returns:
            out: Document의 목록
        """
        out: List[Document] = []
        loader = PyMuPDFLoader(pdf_path, extract_tables="markdown")
        documents = loader.load()

        for idx, doc in enumerate(documents):
            table = self._split_md_tables(doc)
            if len(table) != 0:
                out.append(
                    Document(
                        page_content=table[0],
                        metadata={
                            "source": str(pdf_path),
                            "type": "table",
                            "page": idx + 1,
                        },
                    )
                )
        return out

    def _extract_image_docs(self, pdf_path: Path) -> List[Document]:
        """
        PDF 파일에서 fitz로 페이지마다 이미지를 추출하여 Document로 변환합니다.
        Args:
            pdf_path: PDF 파일 경로
        Returns:
            out: Document의 목록
        """
        out: List[Document] = []

        with fitz.open(pdf_path) as doc:
            end = (
                min(doc.page_count, self.cfg.max_pages)
                if self.cfg.max_pages
                else doc.page_count
            )

            for i in range(end):
                page = doc.load_page(i)
                images = page.get_images(full=True)

                for j, img in enumerate(images):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    img_bytes = base["image"]
                    ext = base.get("ext", "png")

                    img_file = (
                        Path(self.ip.image_output_dir)
                        / f"{pdf_path.stem}"
                        / f"{pdf_path.stem}_p{i+1}_img{j+1}.{ext}"
                    )
                    img_file.parent.mkdir(parents=True, exist_ok=True)
                    if not img_file.exists():
                        img_file.write_bytes(img_bytes)

                    # ✅ 핵심: 이미지 파일 → 캡션(한국어) Document 생성
                    out.extend(
                        self.image_to_docs.make_docs_from_image(
                            image_path=img_file,
                            source=str(pdf_path),
                            page=i + 1,
                            extra_meta={"image_index": j + 1},
                        )
                    )

        return out
