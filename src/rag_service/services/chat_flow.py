# src/rag_service/services/chat_flow.py
from src.rag_service.core.memory import ChatMemory


class ChatService:
    """
    RAG(검색 증강 생성) 기반의 챗봇 서비스 클래스입니다.

    주요 역할:
    1. 사용자 질문 수신
    2. 이전 대화 기록(Memory) 조회 및 포매팅
    3. LLM/Chain에 질문과 문맥 전달 (RAG 실행)
    4. 생성된 답변을 반환하고 메모리에 저장
    """

    def __init__(self, chain):
        """
        Args:
            chain: 랭체인(LangChain)의 실행 체인 (RetrievalQA 등)
        """
        self.chain = chain  # 외부에서 주입받은 RAG 체인 (검색+생성 로직 포함)

        # ChatMemory: 대화 문맥 유지를 위한 메모리 객체
        # max_turns=2: 토큰 제한 및 비용 절약을 위해 최근 2턴(질문-답변 2세트)만 기억합니다.
        self.memory = ChatMemory(max_turns=2)

    def generate_reply(self, user_query: str) -> str:
        """
        사용자 질문을 처리하여 AI 답변을 생성합니다.

        [처리 과정]
        기억 조회 -> 텍스트 전처리(태그 부착) -> Chain 실행 -> 결과 저장 -> 반환

        Args:
            user_query (str): 사용자가 입력한 질문
        Returns:
            str: AI가 생성한 답변
        """

        # [DEBUG 1] 입력 데이터 확인 (개발 단계용 로그)
        print("\n" + "=" * 30)
        print(f"=== [DEBUG] 1. 사용자 질문 입력 확인: {user_query}")

        # 체인이 초기화되지 않았을 경우 방어 코드
        if self.chain is None:
            return "모델이 준비되지 않았습니다."

        # ---------------------------------------------------------
        # 1. [기억 조회] 이전 대화 내용(History)을 메모리에서 가져옵니다.
        # ---------------------------------------------------------
        raw_history = self.memory.get_context()

        # [DEBUG 2] 가져온 메모리 원본 확인 (여기에 현재 질문이 섞여 있으면 안 됨)
        print(f"=== [DEBUG] 2. 메모리(History) 원본: {raw_history}")

        # ---------------------------------------------------------
        # 2. [History 전처리] 대화 내역 포매팅 (Context 오염 방지)
        # ---------------------------------------------------------
        # 줄바꿈(\n) 이슈 및 문맥 혼동을 막기 위해 명시적인 태그(<PREVIOUS_CHAT>)를 사용합니다.
        # 이렇게 하면 AI가 어디까지가 '과거 기억'인지 명확히 구분할 수 있습니다.
        if raw_history and raw_history.strip():
            clean_history = raw_history.replace(
                "\n", " "
            )  # 줄바꿈을 공백으로 치환 (포맷 에러 방지)
            formatted_history = f"<PREVIOUS_CHAT> {clean_history} </PREVIOUS_CHAT>"
        else:
            formatted_history = "대화 기록 없음"

        # ---------------------------------------------------------
        # 3. [Input 전처리] 현재 질문 포매팅
        # ---------------------------------------------------------
        # 질문 내용도 태그(<USER_QUESTION>)로 감싸서, History와 확실히 분리합니다.
        formatted_query = f"<USER_QUESTION> {user_query} </USER_QUESTION>"

        # ---------------------------------------------------------
        # 4. [Payload 구성] Chain에 전달할 데이터 조립
        # ---------------------------------------------------------
        # chat_history: 과거 대화 (배경지식으로 참고)
        # input: 실제 검색하고 답변해야 할 현재 질문
        request_payload = {"chat_history": formatted_history, "input": formatted_query}

        # [DEBUG 3] 최종 입력 데이터 확인 (중요: History와 Input이 분리되었는지 확인)
        print(f"=== [DEBUG] 3. Chain 입력 Payload 확인 ===")
        print(f"   ㄴ chat_history: {request_payload['chat_history']}")
        print(f"   ㄴ input       : {request_payload['input']}")
        print("=" * 30 + "\n")

        # ---------------------------------------------------------
        # 5. [실행] RAG Chain 호출 (검색 -> 프롬프트 결합 -> LLM 생성)
        # ---------------------------------------------------------
        # retrieval.py의 retrieve_multi 함수가 이 단계에서 호출되어 문서를 검색합니다.
        response = self.chain.invoke(request_payload)

        # ---------------------------------------------------------
        # 6. [후처리] 결과 추출
        # ---------------------------------------------------------
        # LangChain 버전에 따라 반환값이 객체(AIMessage)일 수도 있고 문자열일 수도 있어 분기 처리
        final_answer = response.content if hasattr(response, "content") else str(response)

        # [DEBUG 4] 저장 전 데이터 확인
        print(f"=== [DEBUG] 4. 메모리 저장 직전: Q={user_query} / A={final_answer[:30]}...")

        # ---------------------------------------------------------
        # 7. [저장] 이번 턴의 대화를 메모리에 기록
        # ---------------------------------------------------------
        # 다음 질문 때 History로 사용하기 위해 저장합니다.
        self.memory.add(user_query, final_answer)

        return final_answer
