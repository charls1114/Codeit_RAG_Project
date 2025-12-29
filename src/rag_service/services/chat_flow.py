# 파일: src/rag_service/services/chat_flow.py
from src.rag_service.core.memory import ChatMemory


class ChatService:
    # __init__: 필요한 도구(검색기, 체인)를 외부에서 받아옵니다 (의존성 주입)
    def __init__(self, retriever, chain):
        self.retriever = retriever  # 1차 검문소 (검색기)
        self.chain = chain  # 최종 답변가 (AI)
        self.memory = ChatMemory(max_turns=2)  # 최근 2턴 기억

    def generate_reply(self, user_query: str) -> str:
        """
        사용자 질문을 받아 RAG 과정을 거쳐 답변을 생성합니다.
        (검색 -> 메모리 결합 -> 생성 -> 저장)
        """
        # 1. AI 모델이 로딩 안 됐으면 바로 리턴
        if self.chain is None:
            return "모델이 준비되지 않았습니다."

        # 2. [검문] 자료가 있는지 DB 찔러보기
        if self.retriever:
            docs = self.retriever.invoke(user_query)
            if not docs:  # 문서가 하나도 안 나오면
                print(f"📭 검색 결과 없음: '{user_query}'")
                return "참고자료를 찾지 못했습니다."  # 여기서 바로 종료

        # 3. [기억] 이전 대화 내용 가져오기
        history_text = self.memory.get_context()

        # 4. [조합] "이전 대화 + 현재 질문"을 합쳐서 프롬프트 생성
        augmented_query = f"[이전 대화]\n{history_text}\n[현재 질문]\n{user_query}"

        # 5. [생성] AI에게 질문 던지기
        response = self.chain.invoke(augmented_query)
        final_answer = str(response)

        # 6. [저장] 이번 질문과 답변을 메모리에 기록
        self.memory.add(user_query, final_answer)

        return final_answer  # 최종 답변 반환
