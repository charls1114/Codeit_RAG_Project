from typing import List

# 대화 내용을 리스트에 넣고 빼는 자료 구조 관리 담당


class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.history: List[str] = []  # 대화를 저장할 빈 리스트 생성
        self.max_turns = max_turns * 2
        # max_turns는 문답 쌍의 개수입니다. Q와 A가 각각 한 줄씩이니 2를 곱합니다.

    def add(self, query: str, answer: str):
        """질문과 답변을 메모리에 저장합니다."""
        self.history.append(f"Q: {query}")  # 질문 저장
        self.history.append(f"A: {answer}")  # 답변 저장

        # [슬라이딩 윈도우 기법]
        # 저장된 대화가 지정된 한도(max_turns)를 넘으면
        if len(self.history) > self.max_turns:
            self.history.pop(0)  # 가장 오래된 질문 삭제
            self.history.pop(0)  # 가장 오래된 답변 삭제 (항상 쌍으로 지움)

    def get_context(self) -> str:
        """현재 저장된 대화 내용을 문자열로 반환합니다."""
        return "\n".join(self.history)  # 줄바꿈(\n)으로 이어 붙임

    def clear(self):
        """메모리를 초기화합니다."""
        self.history = []
