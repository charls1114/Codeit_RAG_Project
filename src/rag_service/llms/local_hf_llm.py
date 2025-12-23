from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from ..config import get_app_config
from dotenv import load_dotenv


def get_local_hf_llm():
    """
    HuggingFacePipeline을 사용하여 로컬 LLM을 가져옵니다.
    Returns:
        HuggingFacePipeline: 로컬 HuggingFace LLM 객체
    """
    load_dotenv()  # 루트 .env 로드
    cfg = get_app_config()
    model_name = cfg.llm.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=cfg.model_api_key)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=cfg.device,
        dtype="auto",
        # LGAI-EXAONE
        token=cfg.model_api_key,
        trust_remote_code=True,
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=cfg.llm.max_new_tokens,
        do_sample=True,
        temperature=cfg.llm.temperature,
        top_p=0.9,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipeline)
    return llm
