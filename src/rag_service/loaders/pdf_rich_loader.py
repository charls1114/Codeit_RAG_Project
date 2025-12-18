from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import pandas as pd
from langchain_core.documents import Document

import camelot

from ..config import get_app_config, PDFRichLoaderConfig
from ..image_processing.image_to_docs import ImageToDocs, ImageToDocsConfig


# @dataclass
# class PDFRichLoaderConfig:
#     extract_tables: bool = True
#     table_flavor: str = "stream"  # stream/lattice
#     max_pages: Optional[int] = None


class PDFRichLoader:
    def __init__(self, cfg: PDFRichLoaderConfig | None = None):
        self.cfg = cfg or PDFRichLoaderConfig()
        self.app_cfg = get_app_config()

        # ✅ image_processing 설정 반영
        ip = self.app_cfg.image_processing
        Path(ip.image_output_dir).mkdir(parents=True, exist_ok=True)

        self.image_to_docs = ImageToDocs(
            ImageToDocsConfig(
                ocr_enabled=ip.ocr.enabled,
                ocr_lang=ip.ocr.lang,
                min_text_len=ip.ocr.min_text_len,
                caption_enabled=ip.caption.enabled,
                caption_model=ip.caption.model,
                caption_prompt_ko=ip.caption.prompt_ko,
            )
        )

    def load(self, pdf_path: str | Path) -> List[Document]:
        pdf_path = Path(pdf_path)
        docs: List[Document] = []

        docs.extend(self._extract_text_docs(pdf_path))

        if self.cfg.extract_tables:
            docs.extend(self._extract_table_docs(pdf_path))

        if self.app_cfg.image_processing.extract_images:
            docs.extend(self._extract_image_docs(pdf_path))

        return docs

    def _extract_text_docs(self, pdf_path: Path) -> List[Document]:
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

    def _extract_table_docs(self, pdf_path: Path) -> List[Document]:
        out: List[Document] = []
        try:
            tables = camelot.read_pdf(
                str(pdf_path), pages="all", flavor=self.cfg.table_flavor
            )
        except Exception as e:
            out.append(
                Document(
                    page_content=f"[TABLE_EXTRACTION_FAILED] {e}",
                    metadata={"source": str(pdf_path), "type": "table_error"},
                )
            )
            return out

        for idx, t in enumerate(tables):
            df: pd.DataFrame = t.df
            if df is None or df.empty:
                continue
            md = df.to_markdown(index=False)
            out.append(
                Document(
                    page_content=md,
                    metadata={
                        "source": str(pdf_path),
                        "type": "table",
                        "table_index": idx,
                        "page": int(t.page) if hasattr(t, "page") else None,
                        "camelot_flavor": self.cfg.table_flavor,
                    },
                )
            )
        return out

    def _extract_image_docs(self, pdf_path: Path) -> List[Document]:
        out: List[Document] = []
        ip = self.app_cfg.image_processing

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
                        Path(ip.image_output_dir)
                        / f"{pdf_path.stem}"
                        / f"{pdf_path.stem}_p{i+1}_img{j+1}.{ext}"
                    )
                    img_file.parent.mkdir(parents=True, exist_ok=True)
                    if img_file.exists():
                        continue
                    else:
                        img_file.write_bytes(img_bytes)

                    # ✅ 핵심: 이미지 파일 → OCR+캡션(한국어) Document 생성
                    out.extend(
                        self.image_to_docs.make_docs_from_image(
                            image_path=img_file,
                            source=str(pdf_path),
                            page=i + 1,
                            extra_meta={"image_index": j + 1},
                        )
                    )

        return out
