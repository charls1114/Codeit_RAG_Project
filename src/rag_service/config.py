from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from dataclasses import dataclass
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 프로젝트 루트 기준 경로 계산 (src/rag_service/config.py -> ... -> 프로젝트 루트)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ENV_PATH = PROJECT_ROOT / ".env"

# .env 로드 (.env에 있는 값들이 os.environ으로 들어감)
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# ---------- 유틸 함수들 ----------


def load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Pydantic 설정 모델들 ----------


@dataclass
class PDFRichLoaderConfig:
    extract_tables: bool = True
    table_flavor: str = "stream"  # stream/lattice
    max_pages: Optional[int] = None


class OCRConfig(BaseModel):
    enabled: bool = True
    engine: str = "tesseract"  # "tesseract" | "paddleocr"
    lang: str = "kor+eng"
    min_text_len: int = 5


class CaptionConfig(BaseModel):
    enabled: bool = True
    backend: str = "openai"  # 현재 패치에서는 openai 위주
    model: str = "gpt-5-mini"
    prompt_ko: str = (
        "이 이미지를 한국어로 간단히 설명해 주세요. "
        "도표/표/흐름도/아키텍처 그림이라면 핵심 구성요소와 의미를 요약해 주세요."
    )


class ImageProcessingConfig(BaseModel):
    extract_images: bool = True
    image_output_dir: str = "/home/public/data/processed/images"
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    caption: CaptionConfig = Field(default_factory=CaptionConfig)


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 150


class RetrievalConfig(BaseModel):
    k_text: int = 3
    k_table: int = 2
    k_image: int = 2


class VectorStoreConfig(BaseModel):
    persist_dir: str = "/home/public/data/chroma_db"
    collection_name: str = "rfp_rag"


class DocumentConfig(BaseModel):
    allowed_extensions: List[str] = Field(default_factory=lambda: [".pdf", ".hwp"])
    loader_backend: str = "pymupdf_hwp"  # pymupdf_hwp 또는 "llamaindex_file"
    loader_config: PDFRichLoaderConfig = Field(default_factory=PDFRichLoaderConfig)


class LLMConfig(BaseModel):
    # 공통
    model_name: str = None
    temperature: float = 0.0
    max_new_tokens: int = 512


class EmbeddingsConfig(BaseModel):
    model_name: str = None


class LangSmithConfig(BaseModel):
    enabled: bool = True
    api_key: Optional[str] = None
    project: str = "rfp-rag-project"


class AppConfig(BaseModel):
    # 어떤 프로파일을 쓸 것인지: local_hf / openai_api
    rag_mode: str = "openai_api"  # 코드 가독성을 위해 동일 값 유지
    model_api_key: Optional[str] = None
    device: Optional[str] = None  # "cuda" / "cpu" 등

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)

    # ✅ 신규
    image_processing: ImageProcessingConfig = Field(
        default_factory=ImageProcessingConfig
    )


# ---------- 설정 로드 및 병합 함수 ----------


def set_config(config: AppConfig) -> AppConfig:
    """
    .env / 환경변수 값을 AppConfig에 반영.
    '민감 정보'거나 자주 바꾸는 값들 위주로 override.
    """

    # 공통: LangSmith
    ls_enabled = os.getenv("LANGCHAIN_TRACING_V2")
    if ls_enabled is not None:
        config.langsmith.enabled = ls_enabled.lower() == "true"

    ls_api_key = os.getenv("LANGCHAIN_API_KEY")
    if ls_api_key:
        config.langsmith.api_key = ls_api_key

    ls_project = os.getenv("LANGCHAIN_PROJECT")
    if ls_project:
        config.langsmith.project = ls_project

    # 문서 로더 백엔드
    loader_backend = os.getenv("DOC_LOADER_BACKEND")
    if loader_backend:
        config.document.loader_backend = loader_backend

    # 벡터스토어 경로
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR")
    if chroma_dir:
        config.vectorstore.persist_dir = chroma_dir

    # 프로파일/모드 전환 (local_hf / openai_api)
    rag_mode = os.getenv("RAG_MODE")
    if rag_mode:
        config.rag_mode = rag_mode

    # HF 관련
    hf_model = os.getenv("HF_MODEL_NAME")
    hf_emb = os.getenv("HF_EMBEDDING_MODEL")
    hf_api_key = os.getenv("HF_API_KEY")
    device = os.getenv("DEVICE")

    # OpenAI 관련
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model = os.getenv("OPENAI_MODEL_NAME")
    openai_emb = os.getenv("OPENAI_EMBEDDING_MODEL")

    base_cfg = load_yaml_if_exists(CONFIG_DIR / "base.yaml")

    config.chunking.chunk_size = base_cfg.get("chunking", {}).get("chunk_size", 1000)
    config.chunking.chunk_overlap = base_cfg.get("chunking", {}).get(
        "chunk_overlap", 150
    )
    config.retrieval.k_text = base_cfg.get("retrieval", {}).get("k_text", 3)
    config.retrieval.k_table = base_cfg.get("retrieval", {}).get("k_table", 2)
    config.retrieval.k_image = base_cfg.get("retrieval", {}).get("k_image", 2)
    config.llm.temperature = base_cfg.get("llm", {}).get("temperature", 0.0)
    config.llm.max_new_tokens = base_cfg.get("llm", {}).get("max_new_tokens", 512)

    # ✅ image_processing
    config.image_processing.extract_images = base_cfg.get("extract_images", True)
    config.image_processing.image_output_dir = base_cfg.get(
        "image_output_dir", "/home/public/data/processed/images"
    )

    # OCR 설정
    config.image_processing.ocr.lang = base_cfg.get("ocr", {}).get("lang", "kor+eng")
    config.image_processing.ocr.min_text_len = base_cfg.get("ocr", {}).get(
        "min_text_len", 5
    )
    config.image_processing.ocr.engine = base_cfg.get("ocr", {}).get(
        "engine", "tesseract"
    )
    config.image_processing.ocr.enabled = base_cfg.get("ocr", {}).get("enabled", True)

    # Caption 설정
    config.image_processing.caption.model = base_cfg.get("caption", {}).get(
        "model", "gpt-5-mini"
    )
    config.image_processing.caption.prompt_ko = base_cfg.get("caption", {}).get(
        "prompt_ko",
        """이 이미지를 한국어로 간단히 설명해 주세요.
        도표/표/흐름도/아키텍처 그림이라면 핵심 구성요소와 의미를 요약해 주세요.""",
    )
    config.image_processing.caption.enabled = base_cfg.get("caption", {}).get(
        "enabled", True
    )

    # 프로파일에 따라 LLM/임베딩 override
    if config.rag_mode == "local_hf":
        if hf_model:
            config.llm.model_name = hf_model
        if device:
            config.device = device
        if hf_emb:
            config.embeddings.model_name = hf_emb
        if hf_api_key:
            config.model_api_key = hf_api_key

    elif config.rag_mode == "openai_api":
        if openai_model:
            config.llm.model_name = openai_model
        if openai_api_key:
            config.model_api_key = openai_api_key
        if openai_emb:
            config.embeddings.model_name = openai_emb

    return config


# ---------- 외부에서 사용하는 진입점 ----------

_config_cache: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    """
    YAML(base + profile) + .env를 합쳐 최종 AppConfig를 반환.
    (한 번 로드한 뒤 캐시)
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config = AppConfig()
    # .env / 환경변수 반영
    config = set_config(config)
    _config_cache = config
    return config
