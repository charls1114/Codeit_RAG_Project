from .base import BaseRFPDocumentLoader
from .pymupdf_hwp_loader import PyMuPDFHwpRFPDocumentLoader
from .llamaindex_loader import LlamaIndexRFPDocumentLoader
from ..config import get_app_config
from .rfp_rich_loader import RFPRichDocumentLoader


def get_document_loader() -> BaseRFPDocumentLoader:
    """
    RFP 문서 로더를 가져옵니다.
    """
    cfg = get_app_config()
    backend = cfg.document.loader_backend.lower()
    print(f"DOC_LOADER_BACKEND: {backend}")
    if backend == "pymupdf_hwp":
        return PyMuPDFHwpRFPDocumentLoader()
    elif backend == "llamaindex_file":
        return LlamaIndexRFPDocumentLoader()
    elif backend == "rfp_rich_loader":
        return RFPRichDocumentLoader(cfg.document.loader_config)
    else:
        raise ValueError(f"지원하지 않는 DOC_LOADER_BACKEND: {backend}")
