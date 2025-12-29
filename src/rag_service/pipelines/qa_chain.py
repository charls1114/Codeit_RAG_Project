from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List
from langchain_core.documents import Document

from ..llms import get_llm
from .retrieval import retrieve_multi


def _format_docs(docs: List[Document]) -> str:
    """
    문서들을 포맷팅하여 메타데이터를 포함한 문자열로 반환합니다.
    Args:
        docs: Document의 목록
    Returns:
        포맷팅된 문자열
    """
    parts = []
    for d in docs:
        m = d.metadata or {}
        header = (
            f"파일 출처: {m.get('source')} | 페이지: {m.get('page')} | 데이터 타입: {m.get('type')}"
        )
        parts.append(header + "\n" + (d.page_content or ""))
    return "\n\n".join(parts)


def build_rag_chain(k_text: int = 4, k_table: int = 3, k_image: int = 3):
    """
    RAG(Retrieval-Augmented Generation) 체인을 구축합니다.
    Args:
        k_text: 텍스트 청크 검색 개수
        k_table: 표 청크 검색 개수
        k_image: 이미지 청크 검색 개수
    Returns:
        LCEL로 구현된 RAG 체인 객체
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
        당신은 기업 및 정부 제안요청서(RFP)를 분석하는 도우미입니다.
        아래 제공된 문서 컨텍스트를 바탕으로 사용자의 질문에 한국어로 답변하세요.
        필요하다면 목록/표 형태로 요약하되, 근거가 없으면 추측하지 말고 "문서에 정보가 없음"이라고 답하세요.
        개인 정보(예: 회사명, 담당자명 등) 또는 민감한 정보를 답변에 포함하지 마세요.
        # 질문:
        {question}

        # 참조 문서:
        {context}
        """
    )
    retriever = RunnableLambda(
        lambda x: retrieve_multi(x, k_text=k_text, k_table=k_table, k_image=k_image)
    )
    format_ctx = RunnableLambda(lambda docs: _format_docs(docs))

    rag_chain = (
        {
            "question": RunnablePassthrough(),
            "context": retriever | format_ctx,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
