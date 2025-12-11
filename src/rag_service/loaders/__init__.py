from .base import BaseRFPDocumentLoader
from .pymupdf_teddynote_loader import PyMuPDFTeddynoteRFPDocumentLoader
from .llamaindex_loader import LlamaIndexRFPDocumentLoader
from ..config import get_app_config


def get_document_loader() -> BaseRFPDocumentLoader:
    cfg = get_app_config()
    backend = cfg.doc_loader_backend.lower()

    if backend == "pymupdf_teddynote":
        return PyMuPDFTeddynoteRFPDocumentLoader()
    elif backend == "llamaindex_file":
        return LlamaIndexRFPDocumentLoader()
    else:
        raise ValueError(f"지원하지 않는 DOC_LOADER_BACKEND: {backend}")
