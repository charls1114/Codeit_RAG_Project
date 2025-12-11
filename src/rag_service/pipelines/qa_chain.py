from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llms import get_llm
from .retrieval import get_retriever


def build_rag_chain(k: int = 5):
    llm = get_llm()
    retriever = get_retriever(k=k)

    prompt = ChatPromptTemplate.from_template(
        """
당신은 기업 및 정부 제안요청서(RFP)를 분석하는 도우미입니다.
아래 제공된 문서 컨텍스트를 바탕으로 사용자의 질문에 한국어로 답변하세요.
필요하다면 목록/표 형태로 요약하되, 근거가 없으면 추측하지 말고 "문서에 정보가 없음"이라고 답하세요.

# 질문:
{question}

# 참조 문서:
{context}
"""
    )

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
