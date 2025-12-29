from pydantic import BaseModel  # 데이터 유효성 검사를 해주는 도구입니다.

# 질문 시 데이터의 양식 정의


class QuestionRequest(BaseModel):  # 사용자가 보낼 JSON 데이터의 형식을 정의합니다.
    """
    API 요청 시 받을 데이터 형식을 정의합니다.
    """

    query: str  # 'query'라는 이름의 필드는 반드시 문자열(str)이어야 합니다.
