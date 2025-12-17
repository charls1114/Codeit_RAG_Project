from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64

from langchain_core.documents import Document
from PIL import Image

# OCR
try:
    import pytesseract

    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False

# OpenAI (LangChain)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import get_app_config


@dataclass
class ImageToDocsConfig:
    ocr_enabled: bool = True
    ocr_lang: str = "kor+eng"
    min_text_len: int = 5

    caption_enabled: bool = True
    caption_model: str = "gpt-5-mini"
    caption_prompt_ko: str = (
        "이 이미지를 한국어로 간단히 설명해 주세요. "
        "도표/표/흐름도/아키텍처 그림이라면 핵심 구성요소(항목/축/범례/단계)와 의미를 요약해 주세요. "
        "RFP 문서 분석에 도움이 되도록 핵심 정보만 정리해 주세요."
    )


class ImageToDocs:
    def __init__(self, cfg: ImageToDocsConfig):
        self.cfg = cfg
        app_cfg = get_app_config()

        self._openai = ChatOpenAI(
            model=cfg.caption_model,
            api_key=app_cfg.model_api_key,  # OPENAI_API_KEY
            temperature=0.0,
        )

    def make_docs_from_image(
        self,
        image_path: str | Path,
        source: str,
        page: Optional[int] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        image_path = Path(image_path)
        meta = {"source": source, "type": "image", "image_path": str(image_path)}
        if page is not None:
            meta["page"] = page
        if extra_meta:
            meta.update(extra_meta)

        ocr_text = self._run_ocr(image_path) if self.cfg.ocr_enabled else ""
        caption_text = (
            self._run_openai_caption_ko(image_path) if self.cfg.caption_enabled else ""
        )

        parts = [f"[IMAGE] {image_path.name}"]
        if ocr_text:
            parts.append("[OCR]\n" + ocr_text)
        if caption_text:
            parts.append("[CAPTION_KO]\n" + caption_text)

        return [Document(page_content="\n\n".join(parts).strip(), metadata=meta)]

    def _run_ocr(self, image_path: Path) -> str:
        if not TESS_AVAILABLE:
            return ""
        img = Image.open(image_path).convert("RGB")
        txt = (pytesseract.image_to_string(img, lang=self.cfg.ocr_lang) or "").strip()
        return txt if len(txt) >= self.cfg.min_text_len else ""

    def _image_to_data_url(self, image_path: Path) -> str:
        ext = image_path.suffix.lower().lstrip(".") or "png"
        mime = "image/png" if ext in ["png"] else "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _run_openai_caption_ko(self, image_path: Path) -> str:
        data_url = self._image_to_data_url(image_path)

        # OpenAI 비전 입력: content blocks (text + image_url) :contentReference[oaicite:2]{index=2}
        msg = HumanMessage(
            content=[
                {"type": "text", "text": self.cfg.caption_prompt_ko},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        resp = self._openai.invoke([msg])
        return (resp.content or "").strip()
