from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, Any

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
    """
    YAML 파일이 존재하면 로드하고, 아니면 빈 dict를 반환합니다.
    Args:
        path: YAML 파일의 경로
    Returns:
        YAML 내용을 담은 dict, 파일이 없으면 빈 dict
    """
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Pydantic 설정 모델들 ----------

# ---------------------------
#       RAG 시스템 설정
# ----------------------------


class CaptionConfig(BaseModel):
    """
    이미지에서 캡션을 추출하여 Document로 변환할 때 사용하는 설정입니다.
    """

    enabled: bool = True
    model: str = "gpt-5-mini"
    prompt_ko: str = (
        "이 그림을 한국어로 500자 이내로 간단히 설명해 주세요."
        "도표/표/흐름도/아키텍처 그림이라면 핵심 구성요소(항목/축/범례/단계)와 의미를 요약해 주세요. "
        "RFP 문서 분석에 도움이 되도록 핵심 정보만 정리해 주세요."
        "RFP 문서와 관련이 없는 그림이라면 설명하지 마세요."
    )


class ImageProcessingConfig(BaseModel):
    """
    이미지 처리 관련 설정
    """

    extract_images: bool = True
    image_output_dir: str = "/home/public/data/processed/images"
    caption: CaptionConfig = Field(default_factory=CaptionConfig)


class MultiModalLoaderConfig(BaseModel):
    """
    로더 설정
    """

    extract_tables: bool = True
    max_pages: Optional[int] = None
    image_processing: ImageProcessingConfig = Field(default_factory=ImageProcessingConfig)


class ChunkingConfig(BaseModel):
    """
    텍스트 스플리터 청킹 관련 설정
    """

    chunk_size: int = 1000
    chunk_overlap: int = 150


class RetrievalConfig(BaseModel):
    """
    검색기 관련 설정
    """

    k_text: int = 3
    k_table: int = 2
    k_image: int = 2


class VectorStoreConfig(BaseModel):
    """
    벡터스토어 관련 설정
    """

    persist_dir: str = "/home/public/data/chroma_db"
    collection_name: str = "rfp_rag"


class LLMConfig(BaseModel):
    """
    LLM 설정
    """

    model_name: str = None
    temperature: float = 0.0
    max_new_tokens: int = 512


class EmbeddingsConfig(BaseModel):
    """
    임베딩 모델 설정
    """

    model_name: str = None


class LangSmithConfig(BaseModel):
    """
    LangSmith 설정
    """

    enabled: bool = True
    api_key: Optional[str] = None
    project: str = "rfp-rag-project"


class AppConfig(BaseModel):
    """
    애플리케이션의 설정을 담은 Pydantic 모델입니다.
    YAML 파일과 .env를 합쳐서 사용합니다.
    """

    rag_mode: str = "openai_api"  # local_hf 또는 openai_api
    model_api_key: Optional[str] = None
    device: Optional[str] = None  # "cuda" 또는 "cpu"

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    loader_config: MultiModalLoaderConfig = Field(default_factory=MultiModalLoaderConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)

    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)


# ---------- 설정 로드 및 병합 함수 ----------


def set_config(config: AppConfig) -> AppConfig:
    """
    .env / 환경변수 값을 AppConfig에 반영합니다.
    Args:
        config: 기본 AppConfig 객체
    Returns:
        AppConfig: .env / 환경변수 값이 반영된 AppConfig 객체
    """
    # LangSmith API 키 설정
    ls_api_key = os.getenv("LANGCHAIN_API_KEY")
    if ls_api_key:
        config.langsmith.api_key = ls_api_key

    # HF 관련
    hf_model = os.getenv("HF_MODEL_NAME")
    hf_emb = os.getenv("HF_EMBEDDING_MODEL")
    hf_api_key = os.getenv("HF_API_KEY")
    device = os.getenv("DEVICE")

    # OpenAI 관련
    openai_model = os.getenv("OPENAI_MODEL_NAME")
    openai_emb = os.getenv("OPENAI_EMBEDDING_MODEL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

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
        if openai_emb:
            config.embeddings.model_name = openai_emb
        if openai_api_key:
            config.model_api_key = openai_api_key

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
    yaml_config = load_yaml_if_exists(CONFIG_DIR / "base.yaml")

    # ---------------------
    # yaml 설정 로드 및 반영
    # ---------------------

    config = AppConfig(**yaml_config)

    # ---------------------
    # .env / 환경변수 반영
    # ---------------------

    config = set_config(config)
    _config_cache = config
    return config
