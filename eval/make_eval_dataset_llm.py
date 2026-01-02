import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

# 텍스트 스플리터(프로젝트에 맞게 교체 가능)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # 구버전 호환
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# 너희 프로젝트의 LLM 생성 함수 재사용(경로는 프로젝트 구조에 맞게 조정)
from src.rag_service.llms import get_llm


# -------------------------
# PDF -> pages text
# -------------------------
def load_pdf_pages(pdf_path: Path, max_pages: int = 30) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    try:
        n = min(len(doc), max_pages)
        for i in range(n):
            text = doc[i].get_text("text") or ""
            text = normalize_text(text)
            if text.strip():
                pages.append(
                    {
                        "page": i + 1,  # 1-indexed
                        "text": text,
                        "source": str(pdf_path),
                        "type": "text",
                    }
                )
    finally:
        doc.close()
    return pages


def normalize_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# -------------------------
# pages -> chunks
# -------------------------
def make_chunks(
    pages: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: List[Dict[str, Any]] = []

    for p in pages:
        splits = splitter.split_text(p["text"])
        for idx, s in enumerate(splits):
            chunks.append(
                {
                    "source": p["source"],
                    "page": p["page"],
                    "type": p["type"],
                    "chunk_index": idx,
                    "content": s,
                }
            )
    return chunks


# -------------------------
# LLM: context-grounded QA generation
# -------------------------
QA_SYSTEM = """당신은 '정부/공공 제안요청서(RFP) 문서'로부터 평가용 질문/정답(QA) 데이터셋을 생성하는 전문가입니다.

규칙(매우 중요):
1) 반드시 제공된 [컨텍스트] 안에서만 질문과 정답을 생성하세요. 컨텍스트에 없는 사실을 단정하면 안 됩니다.
2) 정답은 컨텍스트에서 그대로 확인 가능한 형태로 간결하게 작성하세요.
3) 문서에 정보가 없거나 불명확하면 정답을 정확히 "문서에 정보가 없음"으로 작성하세요.
4) 아래 4개 핵심 필드에 관한 질문을 우선적으로 생성하세요:
   - 주요 요구 조건, 대상 기관, 예산, 제출 방식
5) 출력은 반드시 JSON 배열만 출력하세요(다른 텍스트 금지).

각 QA 오브젝트 스키마:
{
  "question": "...",
  "reference_answer": "...",
  "target_field": "notice_number|notice_round|project_name|budget|agency|publish_date|bid_start_date|bid_end_date|other",
  "evidence": "정답을 뒷받침하는 컨텍스트 내 문장(짧게)",
  "difficulty": "easy|medium|hard"
}
"""

QA_USER_TEMPLATE = """[컨텍스트]
{context}

[요청]
- 위 컨텍스트만을 근거로 QA를 {n_pairs}개 생성하세요.
- 4개 핵심 필드 질문이 최대한 포함되도록 하세요.
- 컨텍스트가 부족한 필드는 "문서에 정보가 없음"을 정답으로 하여 '거절/보류' 케이스도 포함하세요.
"""


def extract_json_array(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    # 완전 JSON 시도
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass

    # 첫 번째 [...] 블록 추출
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        raise ValueError("LLM output에서 JSON 배열을 찾지 못했습니다.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, list):
        raise ValueError("JSON 배열 형식이 아닙니다.")
    return obj


def generate_qas_for_chunk(
    llm, context: str, n_pairs: int = 3, max_retry: int = 2
) -> List[Dict[str, Any]]:
    prompt = (
        QA_SYSTEM + "\n\n" + QA_USER_TEMPLATE.format(context=context, n_pairs=n_pairs)
    )
    last_err = None

    for _ in range(max_retry + 1):
        out = llm.invoke(prompt)
        text = getattr(out, "content", out)

        try:
            qas = extract_json_array(text)
            # 최소 검증
            cleaned = []
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                if "question" not in qa or "reference_answer" not in qa:
                    continue
                cleaned.append(qa)
            if cleaned:
                return cleaned
            raise ValueError("유효한 QA가 비어있습니다.")
        except Exception as e:
            last_err = e
            prompt = (
                prompt
                + "\n\n[중요] JSON 배열만 출력하세요. 다른 텍스트를 출력하지 마세요."
            )
            continue

    raise RuntimeError(f"QA 생성 실패: {last_err}")


# -------------------------
# dataset writer
# -------------------------
def write_jsonl(items: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="원본 PDF 폴더")
    ap.add_argument("--out", required=True, help="출력 jsonl 경로")
    ap.add_argument("--max_pages", type=int, default=30)
    ap.add_argument("--chunk_size", type=int, default=1200)
    ap.add_argument("--chunk_overlap", type=int, default=150)
    ap.add_argument("--pairs_per_chunk", type=int, default=3)
    ap.add_argument(
        "--max_chunks_per_pdf",
        type=int,
        default=20,
        help="PDF당 최대 몇 청크만 사용할지(비용/시간 제어)",
    )
    ap.add_argument(
        "--min_chars",
        type=int,
        default=400,
        help="컨텍스트 최소 길이(너무 짧은 청크는 제외)",
    )
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {pdf_dir}")

    llm = get_llm()

    dataset: List[Dict[str, Any]] = []

    for pdf_idx, pdf_path in enumerate(pdfs, 1):
        pages = load_pdf_pages(pdf_path, max_pages=args.max_pages)
        chunks = make_chunks(
            pages, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )

        # 너무 짧은 청크 제거
        chunks = [c for c in chunks if len(c["content"]) >= args.min_chars]

        # PDF당 청크 수 제한
        chunks = chunks[: args.max_chunks_per_pdf]

        for c_idx, ch in enumerate(chunks, 1):
            context = (
                f"(출처: {ch['source']} | 페이지: {ch['page']} | 타입: {ch['type']} | 청크: {ch['chunk_index']})\n"
                f"{ch['content']}"
            )

            try:
                qas = generate_qas_for_chunk(
                    llm, context=context, n_pairs=args.pairs_per_chunk
                )
            except Exception as e:
                # 실패해도 진행 (로버스트)
                qas = [
                    {
                        "question": "이 컨텍스트에서 확인 가능한 핵심 정보를 요약해줘",
                        "reference_answer": "문서에 정보가 없음",
                        "target_field": "other",
                        "evidence": f"qa_generation_error: {e}",
                        "difficulty": "hard",
                    }
                ]

            for qa in qas:
                item = {
                    "question": str(qa.get("question", "")).strip(),
                    "reference_answer": str(qa.get("reference_answer", "")).strip(),
                    "metadata": {
                        "pdf_path": ch["source"],
                        "page": ch["page"],
                        "type": ch["type"],
                        "chunk_index": ch["chunk_index"],
                        "target_field": qa.get("target_field", "other"),
                        # Judge에서 근거 확인 가능하도록 "evidence" 저장
                        "evidence": qa.get("evidence", ""),
                        "difficulty": qa.get("difficulty", "medium"),
                    },
                }
                # 빈 질문 방지
                if item["question"]:
                    dataset.append(item)

        print(
            f"[{pdf_idx}/{len(pdfs)}] {pdf_path.name} -> chunks={len(chunks)}, qa_rows_so_far={len(dataset)}"
        )

    write_jsonl(dataset, Path(args.out))
    print(f"✅ 저장 완료: {args.out} (rows={len(dataset)})")


if __name__ == "__main__":
    main()
