import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# 프로젝트 import (경로는 필요시 조정)
# -----------------------------
# 예: 프로젝트 루트에서 실행한다면 `PYTHONPATH=.` 권장
from src.rag_service.pipelines.qa_chain import build_rag_chain  # LCEL RAG 체인
from src.rag_service.pipelines.retrieval import retrieve_multi  # docs retrieval
from src.rag_service.llms import get_llm  # 생성 LLM (프로젝트에서 이미 쓰는 것)


# -----------------------------
# 유틸: docs -> 컨텍스트 문자열 (qa_chain._format_docs와 동일한 스타일)
# -----------------------------
def format_docs_for_judge(docs) -> str:
    parts = []
    for d in docs:
        m = d.metadata or {}
        header = (
            f"파일 출처: {m.get('source')} | 페이지: {m.get('page')} | 데이터 타입: {m.get('type')}"
        )
        parts.append(header + "\n" + (getattr(d, "page_content", "") or ""))
    return "\n\n".join(parts)


# -----------------------------
# 데이터셋 로더: JSONL / CSV
# -----------------------------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    if p.suffix.lower() == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error at line {line_no}: {e}")
                rows.append(obj)
        return rows

    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    raise ValueError("지원 포맷: .jsonl 또는 .csv")


# -----------------------------
# Judge: 프롬프트 + JSON 파서(리트라이)
# -----------------------------
JUDGE_SYSTEM = """당신은 RAG 시스템 평가자(LLM Judge)입니다.
다음 입력(질문/답변/참조 컨텍스트)을 바탕으로, 답변의 품질을 평가하세요.

채점 기준(0~5, 정수):
- correctness: 사실/요구사항 관점에서 정답에 가까운가 (컨텍스트에 근거해 판단)
- faithfulness: 컨텍스트에 없는 내용을 단정/추측하지 않았는가 (환각 여부)
- completeness: 질문이 요구한 항목을 빠짐없이 답했는가
- format_safety: 불필요한 민감정보 노출 없이, '문서에 정보가 없음' 같은 보류가 적절한가

규칙:
- 컨텍스트 밖의 단정이 있으면 faithfulness를 크게 감점하세요.
- 컨텍스트가 부족하면 correctness/completeness는 낮게, format_safety는 높게 줄 수 있습니다.
- 반드시 아래 JSON 스키마로만 출력하세요. 다른 텍스트 출력 금지.

출력 JSON 스키마:
{
  "correctness": 0..5,
  "faithfulness": 0..5,
  "completeness": 0..5,
  "format_safety": 0..5,
  "final": "PASS" | "FAIL",
  "issues": [문제점 문자열...],
  "rationale": "2~4문장"
}
"""

JUDGE_USER_TEMPLATE = """[질문]
{question}

[시스템 답변]
{answer}

[참조 컨텍스트]
{context}
"""


def extract_json(text: str) -> Dict[str, Any]:
    """
    모델이 JSON 외 텍스트를 섞어 출력하는 경우를 대비한 견고 파서.
    """
    text = text.strip()
    # 1) 완전 JSON 시도
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 첫 번째 {...} 블록 추출
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in judge output.")
    return json.loads(m.group(0))


def run_judge(
    judge_llm, question: str, answer: str, context: str, max_retry: int = 2
) -> Dict[str, Any]:
    prompt = (
        JUDGE_SYSTEM
        + "\n\n"
        + JUDGE_USER_TEMPLATE.format(question=question, answer=answer, context=context)
    )

    last_err = None
    for _ in range(max_retry + 1):
        out = judge_llm.invoke(prompt)
        # get_llm()이 ChatModel이면 보통 out.content 형태, 아니면 문자열일 수 있음
        text = getattr(out, "content", out)
        try:
            data = extract_json(text)
            # 최소 필드 체크
            for k in [
                "correctness",
                "faithfulness",
                "completeness",
                "format_safety",
                "final",
                "issues",
                "rationale",
            ]:
                if k not in data:
                    raise ValueError(f"Missing key: {k}")
            return data
        except Exception as e:
            last_err = e
            # 재시도: JSON만 내라고 더 강하게
            prompt = prompt + "\n\n[중요] JSON만 출력하세요. 다른 텍스트 금지."
            continue
    raise RuntimeError(f"Judge parsing failed: {last_err}")


# -----------------------------
# 결과 row 스키마
# -----------------------------
@dataclass
class EvalRow:
    idx: int
    question: str
    reference_answer: str
    answer: str
    k_text: int
    k_table: int
    k_image: int
    n_context_docs: int
    correctness: int
    faithfulness: int
    completeness: int
    format_safety: int
    final: str
    issues: str
    rationale: str


def write_csv(rows: List[EvalRow], out_path: str, add_summary: bool = True) -> None:
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        list(asdict(rows[0]).keys())
        if rows
        else [
            "idx",
            "question",
            "reference_answer",
            "answer",
            "k_text",
            "k_table",
            "k_image",
            "n_context_docs",
            "correctness",
            "faithfulness",
            "completeness",
            "format_safety",
            "final",
            "issues",
            "rationale",
        ]
    )

    with out_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

        if add_summary and rows:
            # 집계 행
            def avg(key: str) -> float:
                return sum(getattr(r, key) for r in rows) / len(rows)

            pass_rate = sum(1 for r in rows if r.final == "PASS") / len(rows)

            w.writerow({})  # 빈 줄
            w.writerow(
                {
                    "idx": "SUMMARY",
                    "question": f"count={len(rows)}",
                    "correctness": f"{avg('correctness'):.3f}",
                    "faithfulness": f"{avg('faithfulness'):.3f}",
                    "completeness": f"{avg('completeness'):.3f}",
                    "format_safety": f"{avg('format_safety'):.3f}",
                    "final": f"pass_rate={pass_rate:.3f}",
                }
            )


# -----------------------------
# 메인
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="QA 데이터셋 경로(.jsonl 또는 .csv)")
    ap.add_argument("--out", required=True, help="결과 CSV 저장 경로")
    ap.add_argument("--k_text", type=int, default=4)
    ap.add_argument("--k_table", type=int, default=3)
    ap.add_argument("--k_image", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체, 아니면 앞에서 N개만 평가")
    ap.add_argument(
        "--judge_model_env", default="JUDGE_MODEL", help="Judge 모델명을 읽을 env var 키"
    )
    ap.add_argument("--add_summary", action="store_true", help="CSV 끝에 요약 행 추가")
    args = ap.parse_args()

    # 1) 데이터셋 로드
    items = load_dataset(args.dataset)
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    # 2) 생성용 RAG 체인 준비
    rag_chain = build_rag_chain(k_text=args.k_text, k_table=args.k_table, k_image=args.k_image)

    # 3) Judge LLM 준비
    # - 프로젝트의 get_llm()을 재사용하되, 환경변수로 Judge 모델을 별도로 지정 가능하게 처리
    # - 예) export JUDGE_MODEL="gpt-4.1" 처럼
    judge_model_name = os.getenv(args.judge_model_env, "").strip()
    if judge_model_name:
        # get_llm()이 내부에서 모델명을 env/config로 받는 구조라면,
        # 아래처럼 임시로 env를 바꿔서 Judge를 만들 수도 있습니다.
        # (프로젝트 get_llm 구현에 맞게 수정 가능)
        os.environ["OPENAI_MODEL"] = judge_model_name  # 프로젝트에서 쓰는 키에 맞춰 바꾸세요
    judge_llm = get_llm()

    rows: List[EvalRow] = []

    for i, obj in enumerate(items):
        q = (obj.get("question") or obj.get("query") or obj.get("Q") or "").strip()
        if not q:
            raise ValueError(f"질문 필드가 비어있습니다. idx={i}, keys={list(obj.keys())}")

        ref = (obj.get("reference_answer") or obj.get("answer") or "").strip()

        # (A) RAG 답변 생성
        try:
            answer = rag_chain.invoke(q)
        except Exception as e:
            answer = f"[ERROR] rag_chain.invoke 실패: {e}"

        # (B) 컨텍스트(검색 문서) 확보 - groundedness 평가용
        try:
            docs = retrieve_multi(q, k_text=args.k_text, k_table=args.k_table, k_image=args.k_image)
            context = format_docs_for_judge(docs)
        except Exception as e:
            docs = []
            context = f"[ERROR] retrieve_multi 실패: {e}"

        # (C) Judge 채점
        try:
            judge = run_judge(judge_llm, question=q, answer=str(answer), context=context)
        except Exception as e:
            judge = {
                "correctness": 0,
                "faithfulness": 0,
                "completeness": 0,
                "format_safety": 0,
                "final": "FAIL",
                "issues": [f"judge_error: {e}"],
                "rationale": "Judge 평가 실패",
            }

        rows.append(
            EvalRow(
                idx=i,
                question=q,
                reference_answer=ref,
                answer=str(answer),
                k_text=args.k_text,
                k_table=args.k_table,
                k_image=args.k_image,
                n_context_docs=len(docs),
                correctness=int(judge["correctness"]),
                faithfulness=int(judge["faithfulness"]),
                completeness=int(judge["completeness"]),
                format_safety=int(judge["format_safety"]),
                final=str(judge["final"]),
                issues=" | ".join(judge.get("issues", [])),
                rationale=str(judge.get("rationale", "")),
            )
        )

        # 진행 표시(가벼운 로깅)
        print(
            f"[{i+1}/{len(items)}] final={rows[-1].final} "
            f"C/F/Comp/S={rows[-1].correctness}/{rows[-1].faithfulness}/{rows[-1].completeness}/{rows[-1].format_safety}"
        )

    # 4) CSV 저장
    write_csv(rows, args.out, add_summary=args.add_summary)
    print(f"✅ 저장 완료: {args.out}")


if __name__ == "__main__":
    main()
