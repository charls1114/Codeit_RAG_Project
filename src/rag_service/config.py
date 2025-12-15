from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Dict, Any

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


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    dict를 재귀적으로 병합하는 함수.
    base에 update를 덮어쓴 결과를 반환.
    """
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------- Pydantic 설정 모델들 ----------


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 150


class RetrievalConfig(BaseModel):
    k: int = 5


class VectorStoreConfig(BaseModel):
    persist_dir: str = str(PROJECT_ROOT / "data" / "chroma_db")
    collection_name: str = "rfp_rag"


class DocumentConfig(BaseModel):
    allowed_extensions: List[str] = Field(default_factory=lambda: [".pdf", ".hwp"])
    loader_backend: str = "pymupdf_hwp"  # pymupdf_hwp 또는 "llamaindex_file"


class LLMConfig(BaseModel):
    # 공통
    model_name: str
    temperature: float = 0.0
    max_new_tokens: int = 512
    api_key: Optional[str] = None

    # HF 전용
    device: Optional[str] = None  # "cuda" / "cpu" 등


class EmbeddingsConfig(BaseModel):
    model_name: str
    api_key: Optional[str] = None  # OpenAI 임베딩용


class LangSmithConfig(BaseModel):
    enabled: bool = True
    api_key: Optional[str] = None
    project: str = "rfp-rag-project"


class AppConfig(BaseModel):
    # 어떤 프로파일을 쓸 것인지: local_hf / openai_api
    profile: str = "openai_api"  # yaml + CONFIG_PROFILE + RAG_MODE로 결정
    rag_mode: str = "openai_api"  # 코드 가독성을 위해 동일 값 유지

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)

    llm: LLMConfig
    embeddings: EmbeddingsConfig

    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)


# ---------- YAML + .env 병합 로직 ----------


def _load_yaml_config_dict() -> Dict[str, Any]:
    """
    base.yaml + {profile}.yaml을 읽어서 병합한 dict 반환.
    profile은 우선순위: ENV(CONFIG_PROFILE) > ENV(RAG_MODE) > local_hf
    """
    env_profile = os.getenv("CONFIG_PROFILE") or os.getenv("RAG_MODE") or "openai_api"
    profile = env_profile.lower()

    base_cfg = load_yaml_if_exists(CONFIG_DIR / "base.yaml")
    profile_cfg = load_yaml_if_exists(CONFIG_DIR / f"{profile}.yaml")

    merged = deep_update(base_cfg, profile_cfg)
    # profile / rag_mode 값은 여기서 명시
    merged.setdefault("profile", profile)
    merged.setdefault("rag_mode", profile)
    return merged


def _apply_env_overrides(config: AppConfig) -> AppConfig:
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
        config.profile = rag_mode
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

    # 프로파일에 따라 LLM/임베딩 override
    if config.rag_mode == "local_hf":
        if hf_model:
            config.llm.model_name = hf_model
        if device:
            config.llm.device = device
        if hf_emb:
            config.embeddings.model_name = hf_emb
        if hf_api_key:
            config.llm.api_key = hf_api_key
    elif config.rag_mode == "openai_api":
        if openai_model:
            config.llm.model_name = openai_model
        if openai_api_key:
            config.llm.api_key = openai_api_key
        if openai_emb:
            config.embeddings.model_name = openai_emb
        if openai_api_key and config.embeddings.api_key is None:
            config.embeddings.api_key = openai_api_key

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

    yaml_dict = _load_yaml_config_dict()
    # YAML dict -> AppConfig
    config = AppConfig(**yaml_dict)
    # .env / 환경변수 반영
    config = _apply_env_overrides(config)
    _config_cache = config
    return config
