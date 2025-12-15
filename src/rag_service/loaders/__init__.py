from .base import BaseRFPDocumentLoader
from .pymupdf_hwp_loader import PyMuPDFHwpRFPDocumentLoader
from .llamaindex_loader import LlamaIndexRFPDocumentLoader
from ..config import get_app_config


def get_document_loader() -> BaseRFPDocumentLoader:
    cfg = get_app_config()
    backend = cfg.document.loader_backend.lower()
    print(f"DOC_LOADER_BACKEND: {backend}")
    if backend == "pymupdf_hwp":
        return PyMuPDFHwpRFPDocumentLoader()
    elif backend == "llamaindex_file":
        return LlamaIndexRFPDocumentLoader()
    else:
        raise ValueError(f"지원하지 않는 DOC_LOADER_BACKEND: {backend}")
