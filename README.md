# Codeit RAG Project (RFP 멀티모달 Q&A)

PDF 제안요청서(RFP) 문서를 **텍스트/표/이미지(캡션)** 단위로 추출 → 청킹/임베딩 → **Chroma 벡터 DB**에 저장하고, 질문이 들어오면 관련 청크를 검색해 **RAG 방식으로 답변**하는 프로젝트입니다.  
`FastAPI`로 간단한 웹 UI(`/`)와 API(`/api/chat`)를 제공합니다.

---

## 주요 기능

- **멀티모달 로딩(PDF)**
  - 텍스트: `PyMuPDF(fitz)`로 페이지별 텍스트 추출
  - 표: `PyMuPDFLoader(extract_tables="markdown")`로 테이블 마크다운 추출 → 정규식으로 표 블록만 분리
  - 이미지: PDF 내 이미지를 저장한 뒤(설정된 폴더), **이미지 캡션 생성** → Document로 적재
- **RAG 파이프라인**
  - 문서 적재: `src/rag_service/pipelines/ingest.py`
  - 검색: 텍스트/표/이미지 타입별 k개 검색
  - 생성: 검색 컨텍스트 기반 답변 생성(근거 없으면 “문서에 정보가 없음”)
- **서빙**
  - 웹 서버: `scripts/run_server.py` (FastAPI + static UI)
  - CLI: `scripts/run_qa_cli.py`
- **옵션**
  - `OpenAI API` 기반 모드 / `로컬 HF` 기반 모드 전환 가능 (`configs/base.yaml`의 `rag_mode`)
  - LangSmith 트레이싱(선택)

---

## 프로젝트 구조

├─ configs/  
│ └─ base.yaml # 기본 설정(로더/청킹/검색 k/LLM 등)  
├─ scripts/  
│ ├─ run_server.py # FastAPI 서버(웹 UI + /api/chat)  
│ └─ run_qa_cli.py # 터미널 Q&A CLI  
├─ src/  
│ └─ rag_service/  
│ ├─ chunking/ # 청킹 로직  
│ ├─ core/ # startup/메모리 등 공통  
│ ├─ embeddings/ # OpenAI/HF 임베딩  
│ ├─ image_processing/ # 이미지→캡션→Document  
│ ├─ llms/ # OpenAI/HF LLM  
│ ├─ loaders/ # 멀티모달 PDF 로더  
│ ├─ models/ # FastAPI 요청 스키마 등  
│ ├─ pipelines/ # ingest/retrieval/qa_chain  
│ ├─ services/ # ChatService(대화 흐름 매니저)  
│ └─ vectorstores/ # Chroma 저장/로드  
├─ static/ # 프론트 정적 리소스(index.html, js, css)  
├─ notebooks/ # 실험/평가 노트북  
├─ requirements.txt  
├─ .env.example  
└─ README.md  


---
## 최종 발표 자료(보고서)
[📄 프로젝트 보고서 (PDF)](team5_report.pdf)

## 개인 협업 일지

이승철 - https://www.notion.so/2c5eaebccce68043bcf4e23d205eb5a6?v=2a0eaebccce6812f8d98000c7791451e&source=copy_link

김동현 -

이경식 - https://www.notion.so/2c53a594a4d08039b8e5c8c73e82fac7?source=copy_link

최경운 - 
