# src/rag_service/services/chat_flow.py
from src.rag_service.core.memory import ChatMemory


class ChatService:
    def __init__(self, chain):
        self.chain = chain
        self.memory = ChatMemory(max_turns=2)

    def generate_reply(self, user_query: str) -> str:
        # [DEBUG 1] 현재 들어온 질문 확인
        print("\n" + "=" * 30)
        print(f"=== [DEBUG] 1. 사용자 질문 입력 확인: {user_query}")

        if self.chain is None:
            return "모델이 준비되지 않았습니다."

        # 1. [기억] 이전 대화 가져오기
        raw_history = self.memory.get_context()

        # [DEBUG 2] 메모리 원본 확인 (여기에 현재 질문이 섞여있으면 안 됨)
        print(f"=== [DEBUG] 2. 메모리(History) 원본: {raw_history}")

        # 2. [전처리] 히스토리 데이터 포매팅
        if raw_history and raw_history.strip():
            clean_history = raw_history.replace("\n", " ")
            formatted_history = f"<PREVIOUS_CHAT> {clean_history} </PREVIOUS_CHAT>"
        else:
            formatted_history = "대화 기록 없음"

        # 3. [전처리] 질문 데이터 포매팅
        formatted_query = f"<USER_QUESTION> {user_query} </USER_QUESTION>"

        # 4. [페이로드 구성]
        request_payload = {"chat_history": formatted_history, "input": formatted_query}

        # [DEBUG 3] 최종 LLM/Chain 입력 데이터 확인 (가장 중요!)
        # 여기서 chat_history와 input이 서로 침범하지 않고 분리되어 있는지 확인
        print(f"=== [DEBUG] 3. Chain 입력 Payload 확인 ===")
        print(f"   ㄴ chat_history: {request_payload['chat_history']}")
        print(f"   ㄴ input       : {request_payload['input']}")
        print("=" * 30 + "\n")

        # 5. [실행]
        response = self.chain.invoke(request_payload)

        # 6. [후처리 및 저장]
        final_answer = response.content if hasattr(response, "content") else str(response)

        # [DEBUG 4] 저장되기 전 데이터 확인
        print(f"=== [DEBUG] 4. 메모리 저장 직전: Q={user_query} / A={final_answer[:30]}...")

        self.memory.add(user_query, final_answer)

        return final_answer
