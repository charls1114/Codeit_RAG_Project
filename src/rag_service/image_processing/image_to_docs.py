from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64
from langchain_core.documents import Document

# OpenAI (LangChain)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import get_app_config


class ImageToDocs:
    """
    이미지에서 OCR 텍스트와 캡션을 추출하여 Document로 변환합니다.
    """

    def __init__(self):
        app_cfg = get_app_config()
        self.caption_cfg = app_cfg.loader_config.image_processing.caption
        self._openai = ChatOpenAI(
            model=app_cfg.loader_config.image_processing.caption.model,
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
        """
        이미지에서 캡션을 추출하여 Document로 변환합니다.
        Args:
            image_path: 이미지 파일 경로
            source: 원본 문서의 소스 정보
            page: 원본 문서의 페이지 번호
            extra_meta: 추가 메타데이터
        Returns:
            List[Document]: Document 목록
        """
        image_path = Path(image_path)
        meta = {"source": source, "type": "image", "image_path": str(image_path)}
        if page is not None:
            meta["page"] = page
        if extra_meta:
            meta.update(extra_meta)

        caption_text = (
            self._run_openai_caption_ko(image_path) if self.caption_cfg.enabled else ""
        )

        parts = [f"[IMAGE] {image_path.name}"]
        if caption_text:
            parts.append("[CAPTION_KO]\n" + caption_text)
        return [Document(page_content="\n\n".join(parts).strip(), metadata=meta)]

    def _image_to_data_url(self, image_path: Path) -> str:
        """
        이미지를 base64 인코딩된 data URL로 변환합니다.
        Args:
            image_path: 이미지 파일 경로
        Returns:
            str: data URL 문자열
        """
        ext = image_path.suffix.lower().lstrip(".") or "png"
        mime = "image/png" if ext in ["png"] else "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _run_openai_caption_ko(self, image_path: Path) -> str:
        """
        OpenAI를 사용하여 이미지에 대한 한국어 캡션을 생성합니다.
        Args:
            image_path: 이미지 파일 경로
        Returns:
            str: 생성된 캡션
        """
        data_url = self._image_to_data_url(image_path)

        # OpenAI 비전 입력: content blocks (text + image_url) :contentReference[oaicite:2]{index=2}
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": self.caption_cfg.prompt_ko,
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        resp = self._openai.invoke([msg])
        return (resp.content or "").strip()
