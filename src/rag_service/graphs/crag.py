# src/rag_service/graphs/crag_graph.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# 1) 필드/근거 스키마
# =========================================================
FIELD_LABELS_KO: Dict[str, str] = {
    "notice_number": "공고 번호",
    "notice_round": "공고 차수",
    "project_name": "사업명",
    "project_budget": "사업 금액",
    "ordering_agency": "발주 기관",
    "publish_date": "공개 일자",
    "bid_start_date": "입찰 참여 시작일",
    "bid_end_date": "입찰 참여 마감일",
}

# 누락 필드별 검색 우선 type
# (문서 특성상 일정/금액은 표에, 기관/사업명은 텍스트에 있을 확률이 높음)
FIELD_TYPE_PRIORITY: Dict[str, List[str]] = {
    "project_budget": ["table", "text", "image"],
    "publish_date": ["table", "text", "image"],
    "bid_start_date": ["table", "text", "image"],
    "bid_end_date": ["table", "text", "image"],
    "notice_number": ["text", "table", "image"],
    "notice_round": ["text", "table", "image"],
    "project_name": ["text", "table", "image"],
    "ordering_agency": ["text", "table", "image"],
}

# 기본 type 순서
DEFAULT_TYPE_ORDER = ["text", "table", "image"]


class EvidenceItem(BaseModel):
    doc_id: int = Field(..., description="컨텍스트 내 DOC 인덱스(0부터)")
    snippet: str = Field(..., description="근거 발췌(최대 200자 권장)")


class FieldValueWithEvidence(BaseModel):
    value: Optional[str] = Field(default=None, description="추출된 값(없으면 null)")
    evidences: List[EvidenceItem] = Field(
        default_factory=list, description="근거(없으면 빈 리스트)"
    )


class RfpRequiredInfoWithEvidence(BaseModel):
    notice_number: FieldValueWithEvidence = Field(
        default_factory=FieldValueWithEvidence
    )
    notice_round: FieldValueWithEvidence = Field(default_factory=FieldValueWithEvidence)
    project_name: FieldValueWithEvidence = Field(default_factory=FieldValueWithEvidence)
    project_budget: FieldValueWithEvidence = Field(
        default_factory=FieldValueWithEvidence
    )
    ordering_agency: FieldValueWithEvidence = Field(
        default_factory=FieldValueWithEvidence
    )
    publish_date: FieldValueWithEvidence = Field(default_factory=FieldValueWithEvidence)
    bid_start_date: FieldValueWithEvidence = Field(
        default_factory=FieldValueWithEvidence
    )
    bid_end_date: FieldValueWithEvidence = Field(default_factory=FieldValueWithEvidence)

    def missing_fields(self) -> List[str]:
        miss = []
        d = self.model_dump()
        for k, v in d.items():
            val = v.get("value")
            if val is None or (isinstance(val, str) and not val.strip()):
                miss.append(k)
        return miss

    def merge_fill_missing(
        self, other: "RfpRequiredInfoWithEvidence"
    ) -> "RfpRequiredInfoWithEvidence":
        """
        self에서 비어있는 value만 other로 채우고, evidences도 함께 채움
        """
        base = self.model_dump()
        new = other.model_dump()
        merged: Dict[str, Any] = {}
        for k in base.keys():
            base_val = base[k]["value"]
            if base_val is None or (isinstance(base_val, str) and not base_val.strip()):
                merged[k] = new[k]
            else:
                # 값이 있으면 evidence는 누적(중복 제거는 나중에)
                merged[k] = {
                    "value": base[k]["value"],
                    "evidences": base[k]["evidences"] + new[k]["evidences"],
                }
        return RfpRequiredInfoWithEvidence(**merged)


class FieldEvidenceResolved(BaseModel):
    value: Optional[str] = None
    evidences: List[Dict[str, Any]] = Field(default_factory=list)
    # evidence item: {doc_id, snippet, source, page, type}


class FinalRequiredInfoPayload(BaseModel):
    notice_number: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    notice_round: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    project_name: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    project_budget: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    ordering_agency: FieldEvidenceResolved = Field(
        default_factory=FieldEvidenceResolved
    )
    publish_date: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    bid_start_date: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)
    bid_end_date: FieldEvidenceResolved = Field(default_factory=FieldEvidenceResolved)

    def missing_fields(self) -> List[str]:
        miss = []
        for k, v in self.model_dump().items():
            val = v.get("value")
            if val is None or (isinstance(val, str) and not val.strip()):
                miss.append(k)
        return miss


# =========================================================
# 2) Graph State
# =========================================================
@dataclass
class CRAGState:
    original_question: str
    question: str

    docs: List[Document] = field(default_factory=list)

    extracted: RfpRequiredInfoWithEvidence = field(
        default_factory=RfpRequiredInfoWithEvidence
    )
    missing: List[str] = field(default_factory=list)

    attempt: int = 0
    max_attempts: int = 2

    # corrective 검색용 쿼리 후보들(필드별 2~3개 생성)
    corrective_queries: List[str] = field(default_factory=list)

    # 내부 기록
    trace: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# 3) Utils: docs 포맷/중복제거
# =========================================================
def _format_docs(docs: List[Document], max_chars: int = 14000) -> str:
    """
    LLM 컨텍스트용 포맷: DOC id가 반드시 들어가도록.
    """
    parts: List[str] = []
    total = 0
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        header = f"[DOC {i}] source={meta.get('source')} page={meta.get('page')} type={meta.get('type')}\n"
        body = (d.page_content or "").strip()
        chunk = header + body + "\n"
        if total + len(chunk) > max_chars:
            remain = max_chars - total
            if remain > 0:
                parts.append(chunk[:remain])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


def _doc_key(d: Document) -> Tuple[str, Any, Any, str]:
    meta = d.metadata or {}
    # source/page/type + content head 일부로 안정적 유니크 키
    content = (d.page_content or "").strip()
    head = content[:180]
    return (str(meta.get("source")), meta.get("page"), meta.get("type"), head)


def _dedup_docs(docs: List[Document]) -> List[Document]:
    seen: Set[Tuple[str, Any, Any, str]] = set()
    out: List[Document] = []
    for d in docs:
        k = _doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


def _filter_docs_by_type(docs: List[Document], typ: str) -> List[Document]:
    return [d for d in docs if (d.metadata or {}).get("type") == typ]


def _resolve_evidences(
    extracted: RfpRequiredInfoWithEvidence,
    docs: List[Document],
) -> FinalRequiredInfoPayload:
    """
    doc_id -> metadata(source/page/type) 붙여서 최종 payload 생성
    """
    docs_meta = []
    for i, d in enumerate(docs):
        meta = d.metadata or {}
        docs_meta.append(
            {
                "doc_id": i,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "type": meta.get("type"),
            }
        )

    def resolve_field(f: FieldValueWithEvidence) -> FieldEvidenceResolved:
        val = f.value
        evids = []
        seen = set()
        for e in f.evidences:
            if e.doc_id < 0 or e.doc_id >= len(docs_meta):
                continue
            m = docs_meta[e.doc_id]
            key = (e.doc_id, (e.snippet or "").strip())
            if key in seen:
                continue
            seen.add(key)
            evids.append(
                {
                    "doc_id": e.doc_id,
                    "snippet": (e.snippet or "").strip()[:240],
                    "source": m["source"],
                    "page": m["page"],
                    "type": m["type"],
                }
            )
        return FieldEvidenceResolved(value=val, evidences=evids)

    data = extracted.model_dump()
    resolved: Dict[str, Any] = {}
    for k in data.keys():
        fv = FieldValueWithEvidence(**data[k])
        resolved[k] = resolve_field(fv).model_dump()
    return FinalRequiredInfoPayload(**resolved)


# =========================================================
# 4) Retrieve 노드들: 기본/타입우선/union 검색
# =========================================================
def make_retrieve_basic_node(retriever, k: Optional[int] = None):
    def _retrieve(state: CRAGState) -> CRAGState:
        if k is not None:
            try:
                retriever.search_kwargs = {
                    **getattr(retriever, "search_kwargs", {}),
                    "k": k,
                }
            except Exception:
                pass

        docs = retriever.get_relevant_documents(state.question)
        docs = _dedup_docs(docs)
        state.docs = docs

        state.trace[f"retrieve_basic_attempt_{state.attempt}"] = {
            "query": state.question,
            "k": k,
            "num_docs": len(docs),
        }
        return state

    return _retrieve


def make_retrieve_type_priority_union_node(retriever, *, k_by_type: Dict[str, int]):
    """
    누락필드 기반으로 type 순서를 결정하고,
    type별로 쿼리 2~3개씩 union 검색.
    (현 retriever가 metadata filter를 지원하지 않는 경우가 많아서,
     여기서는 type별 '별도 검색'이 아니라 '검색 후 type으로 재정렬/상위 선별' 방식 + 필요 시 k 확대를 사용)
    """

    def _retrieve(state: CRAGState) -> CRAGState:
        # 1) type 우선 순서 결정
        missing = state.missing or []
        type_order = DEFAULT_TYPE_ORDER[:]
        if missing:
            # 가장 “중요하게 누락된” 필드 하나 기준으로 우선순위를 정함(간단 전략)
            primary = missing[0]
            type_order = FIELD_TYPE_PRIORITY.get(primary, DEFAULT_TYPE_ORDER)

        # 2) corrective queries를 사용해 union 검색
        union_docs: List[Document] = []
        for q in state.corrective_queries:
            docs = retriever.get_relevant_documents(q)
            union_docs.extend(docs)

        union_docs = _dedup_docs(union_docs)

        # 3) type 우선순서로 재정렬 + type별 상위 k 선별
        selected: List[Document] = []
        seen_keys: Set[Tuple[str, Any, Any, str]] = set()

        for typ in type_order:
            cand = [d for d in union_docs if (d.metadata or {}).get("type") == typ]
            limit = k_by_type.get(typ, 0)
            for d in cand[:limit]:
                kkey = _doc_key(d)
                if kkey in seen_keys:
                    continue
                seen_keys.add(kkey)
                selected.append(d)

        # 남는 문서가 너무 적으면(파편화) union_docs에서 추가 보충
        if len(selected) < sum(k_by_type.values()):
            for d in union_docs:
                kkey = _doc_key(d)
                if kkey in seen_keys:
                    continue
                seen_keys.add(kkey)
                selected.append(d)
                if len(selected) >= sum(k_by_type.values()):
                    break

        state.docs = selected

        state.trace[f"retrieve_type_union_attempt_{state.attempt}"] = {
            "missing": missing,
            "type_order": type_order,
            "queries": state.corrective_queries,
            "k_by_type": k_by_type,
            "union_num_docs": len(union_docs),
            "selected_num_docs": len(selected),
        }
        return state

    return _retrieve


# =========================================================
# 5) Extract 노드: 8필드 + DOC 근거(발췌)까지
# =========================================================
def make_extract_node(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 제안요청서(RFP)에서 필수 메타정보 8종을 추출하는 정보추출기다.\n"
                "- 반드시 제공된 '문서 컨텍스트'에서만 근거를 찾아라.\n"
                "- 값이 없으면 null로 둔다.\n"
                "- 각 필드마다 근거가 되는 DOC id(정수)와 발췌 snippet(짧게)을 1~2개 제공하라.\n"
                "- snippet은 해당 값이 직접 보이는 부분을 200자 이내로 포함해라.\n"
                "- 추정/상상/일반상식으로 채우기 금지.",
            ),
            (
                "user",
                "질문: {question}\n\n"
                "아래 문서 컨텍스트에서 필수정보 8종을 추출해라.\n"
                "필드:\n"
                "- notice_number(공고 번호)\n"
                "- notice_round(공고 차수)\n"
                "- project_name(사업명)\n"
                "- project_budget(사업 금액)\n"
                "- ordering_agency(발주 기관)\n"
                "- publish_date(공개 일자)\n"
                "- bid_start_date(입찰 참여 시작일)\n"
                "- bid_end_date(입찰 참여 마감일)\n\n"
                "문서 컨텍스트:\n{context}",
            ),
        ]
    )

    structured_llm = llm.with_structured_output(RfpRequiredInfoWithEvidence)

    def _extract(state: CRAGState) -> CRAGState:
        ctx = _format_docs(state.docs)
        extracted: RfpRequiredInfoWithEvidence = structured_llm.invoke(
            prompt.format_messages(question=state.question, context=ctx)
        )

        state.extracted = extracted
        state.missing = extracted.missing_fields()

        state.trace[f"extract_attempt_{state.attempt}"] = {
            "missing": state.missing,
            "extracted_values": {
                k: v["value"] for k, v in extracted.model_dump().items()
            },
        }
        return state

    return _extract


# =========================================================
# 6) Corrective: 누락 필드별 2~3개 쿼리 생성 노드
# =========================================================
def make_build_corrective_queries_node(llm: BaseChatModel, *, per_field: int = 2):
    """
    누락 필드마다 검색용 질의를 2개 생성 -> 합쳐서 2~3개 정도로 압축(중복 제거)
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 RFP 문서에서 특정 필드를 찾기 위한 검색 질의를 생성하는 도우미다.\n"
                '출력은 JSON 배열 문자열(예: ["...","..."])로만 출력해라.\n'
                "질의는 짧고 키워드 중심으로.\n"
                "한국 문서에서 쓰는 동의어/표기 변형을 포함해라.",
            ),
            (
                "user",
                "원 질문: {question}\n"
                "누락 필드: {missing_fields}\n\n"
                "누락 필드를 찾기 위한 검색 질의들을 {n}개 이상 생성해라.\n"
                "필드별 힌트:\n"
                "- 공고 번호: 공고번호/공고문번호/입찰공고번호/공고 No\n"
                "- 공고 차수: 공고차수/차수/1차/2차/정정공고\n"
                "- 사업명: 사업명/용역명/과업명/프로젝트명\n"
                "- 사업 금액: 사업금액/추정가격/예정가격/기초금액/금액(원)/budget\n"
                "- 발주 기관: 발주기관/수요기관/사업자(기관)/기관명\n"
                "- 공개 일자: 공고일/게시일/공개일/작성일\n"
                "- 입찰 참여 시작일: 접수시작/제안서 접수 시작/투찰시작/입찰 시작\n"
                "- 입찰 참여 마감일: 접수마감/제안서 접수 마감/투찰마감/입찰 마감",
            ),
        ]
    )

    def _node(state: CRAGState) -> CRAGState:
        missing = state.missing or []
        # 누락이 없으면 비워둠
        if not missing:
            state.corrective_queries = []
            return state

        n = max(per_field, 2)
        msg = prompt.format_messages(
            question=state.original_question,
            missing_fields=", ".join([FIELD_LABELS_KO.get(f, f) for f in missing]),
            n=str(n),
        )
        raw = llm.invoke(msg).content.strip()

        # 안전 파싱(LLM이 JSON을 살짝 깨뜨릴 수 있으므로 최소 보정)
        queries: List[str] = []
        try:
            import json

            queries = json.loads(raw)
            if not isinstance(queries, list):
                queries = []
        except Exception:
            # fallback: 줄바꿈/콤마 분리
            raw2 = raw.strip().strip("[]")
            queries = [
                q.strip().strip('"').strip("'") for q in raw2.split(",") if q.strip()
            ]

        # 원질문도 포함(Recall 강화)
        merged = [state.original_question] + [q for q in queries if q and q.strip()]

        # 중복 제거 + 너무 긴 쿼리 절단
        seen = set()
        final_qs = []
        for q in merged:
            qn = " ".join(q.split())
            if not qn:
                continue
            if qn in seen:
                continue
            seen.add(qn)
            final_qs.append(qn[:220])

        # 너무 많으면 상위 3~5개로 제한
        state.corrective_queries = final_qs[:5]

        state.trace[f"build_corrective_queries_attempt_{state.attempt}"] = {
            "missing": missing,
            "corrective_queries": state.corrective_queries,
            "raw": raw[:500],
        }
        return state

    return _node


# =========================================================
# 7) Merge 노드: 새 추출값을 누적 + evidence 누적
# =========================================================
def make_merge_node(llm: BaseChatModel):
    extract_node = make_extract_node(llm)

    def _merge(state: CRAGState) -> CRAGState:
        prev = state.extracted
        state = extract_node(state)
        state.extracted = prev.merge_fill_missing(state.extracted)
        state.missing = state.extracted.missing_fields()

        state.trace[f"merge_attempt_{state.attempt}"] = {
            "missing_after_merge": state.missing,
            "merged_values": {
                k: v["value"] for k, v in state.extracted.model_dump().items()
            },
        }
        return state

    return _merge


# =========================================================
# 8) 라우팅
# =========================================================
def route_should_correct(state: CRAGState) -> str:
    if state.missing and state.attempt < state.max_attempts:
        return "correct"
    return "final"


# =========================================================
# 9) 최종 답변 노드: 표 + 누락 표시 + 근거 포함
# =========================================================
def make_answer_node(llm: BaseChatModel):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 제안요청서(RFP) 정보 제공 도우미다.\n"
                "필수항목 8종이 '검색/추출되었는지'를 표로 정리하고, 각 항목에 대해 근거(DOC 메타데이터)를 가능한 한 제시하라.\n"
                "값이 없으면 '문서에서 확인 불가'로 표시하라.\n"
                "표 아래에 3~6문장으로 핵심 요약을 작성하라.",
            ),
            (
                "user",
                "사용자 질문: {question}\n\n"
                "필수정보+근거(JSON, doc_id 기반):\n{payload}\n\n"
                "DOC 컨텍스트(참고):\n{context}\n\n"
                "출력 형식:\n"
                "1) Markdown 표(항목/값/근거(source,page,type,doc_id))\n"
                "2) 핵심 요약(3~6문장)\n",
            ),
        ]
    )

    def _answer(state: CRAGState) -> Dict[str, Any]:
        payload = _resolve_evidences(state.extracted, state.docs)
        ctx = _format_docs(state.docs)

        msg = prompt.format_messages(
            question=state.original_question,
            payload=str(payload.model_dump()),
            context=ctx,
        )
        out = llm.invoke(msg).content

        return {
            "answer": out,
            # “검색되었는가?” 판단용으로 필드별 근거까지 포함한 payload 제공
            "required_info": payload.model_dump(),
            "missing_fields": payload.missing_fields(),
            "debug": {
                "attempts": state.attempt,
                "trace": state.trace,
            },
        }

    return _answer


# =========================================================
# 10) Graph build
# =========================================================
def build_crag_graph(
    llm: BaseChatModel,
    retriever,
    *,
    max_attempts: int = 2,
    k_first: int = 8,
    # corrective 단계: type별로 몇 개씩 확보할지
    k_by_type_corrective: Optional[Dict[str, int]] = None,
    per_field_queries: int = 2,
):
    if k_by_type_corrective is None:
        k_by_type_corrective = {"text": 4, "table": 6, "image": 2}

    graph = StateGraph(CRAGState)

    # nodes
    retrieve_first = make_retrieve_basic_node(retriever, k=k_first)
    extract = make_extract_node(llm)

    build_queries = make_build_corrective_queries_node(llm, per_field=per_field_queries)
    retrieve_correct = make_retrieve_type_priority_union_node(
        retriever, k_by_type=k_by_type_corrective
    )
    merge = make_merge_node(llm)

    def _inc_attempt(state: CRAGState) -> CRAGState:
        state.attempt += 1
        return state

    answer = make_answer_node(llm)

    graph.add_node("retrieve_first", retrieve_first)
    graph.add_node("extract", extract)
    graph.add_node("build_queries", build_queries)
    graph.add_node("retrieve_correct", retrieve_correct)
    graph.add_node("merge", merge)
    graph.add_node("inc_attempt", _inc_attempt)
    graph.add_node("answer", answer)

    # edges
    graph.set_entry_point("retrieve_first")
    graph.add_edge("retrieve_first", "extract")

    graph.add_conditional_edges(
        "extract", route_should_correct, {"correct": "build_queries", "final": "answer"}
    )

    graph.add_edge("build_queries", "retrieve_correct")
    graph.add_edge("retrieve_correct", "merge")
    graph.add_edge("merge", "inc_attempt")
    graph.add_edge("inc_attempt", "extract")

    graph.add_edge("answer", END)

    compiled = graph.compile()

    def run(question: str) -> Dict[str, Any]:
        init = CRAGState(
            original_question=question,
            question=question,
            max_attempts=max_attempts,
        )
        return compiled.invoke(init)

    return compiled, run
