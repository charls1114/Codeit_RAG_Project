from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from ..config import get_app_config


def get_local_hf_llm():
    cfg = get_app_config()
    model_name = cfg.llm.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=cfg.llm.device,
        torch_dtype="auto",
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
