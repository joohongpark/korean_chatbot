# 한국어 학습자 오류 분석 및 쓰기 피드백 시스템

한국어 학습자 말뭉치를 기반으로 오류 문장을 검색·분석하는 연구 프로젝트와,
분석 결과를 활용해 실시간 쓰기 피드백을 제공하는 챗봇으로 구성됩니다.

---

## 프로젝트 구조

```
.
├── 가설 1.ipynb          # BM25 / 벡터 / 하이브리드 검색 비교 실험
├── 가설 2.ipynb          # 형태소 분석 및 가중치 N-gram 유사도 실험
├── 가설 3.ipynb          # 오류 주석 활용 검색 정확도 개선 실험
├── 가설 4.ipynb          # LLM 기반 RAG 파이프라인 실험
├── 인덱스 구축.ipynb      # OpenSearch 인덱스 빌드 스크립트
├── opensearch.ipynb      # OpenSearch 쿼리 테스트
├── 말뭉치.parquet.gzip   # 한국어 학습자 오류 말뭉치
└── chatbot/              # 피드백 챗봇 서버
    ├── app.py            # FastAPI 애플리케이션
    ├── rag.py            # OpenSearch 하이브리드 검색 + 포매팅
    ├── db.py             # SQLite 영속 저장 (users / conversations / turns)
    ├── 프롬프트.json       # Gemini 프롬프트 설정
    └── static/
        ├── index.html    # 학습자용 챗봇 UI
        └── admin.html    # 어드민 대시보드
```

---

## 연구 파이프라인

### 말뭉치 및 인덱스

- **데이터**: 학습자 오류 말뭉치 235,902문장 (오류 주석 65,791건)
- **OpenSearch 인덱스**: BM25(nori 분석기) + kNN 벡터(HNSW cosine, 1024차원)
- **임베딩 모델**: [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1) (로컬 `model/KURE-v1/`)
- **형태소 분석**: `kiwipiepy`

### 실험 가설

| 노트북 | 내용 | 결과 요약 |
|--------|------|-----------|
| 가설 1 | BM25 / 벡터 / 하이브리드 검색 비교 | 유사 문장은 찾지만 유사 오류는 미흡 |
| 가설 2 | 형태소 N-gram 가중치 유사도 | 문장 수준 유사도 개선, 오류 검색 한계 동일 |
| 가설 3 | 오류 시그니처·교정 쌍 직접 검색 | 오류 유형을 알 때 정밀도 매우 높음 |
| 가설 4 | LLM이 오류 진단 후 말뭉치 RAG 검색 | 진단 결과를 챗봇 피드백에 활용 |

---

## 챗봇 서버

### 기술 스택

- **백엔드**: FastAPI + Google Gemini 2.5 Flash
- **RAG**: OpenSearch 하이브리드 검색 → 유사 오류 사례 자연어 포매팅
- **저장소**: SQLite (WAL 모드, `chatbot/chatbot.db`)
- **인증**: Bearer 토큰 (username + 비밀번호)

### 실행 방법

```bash
# 의존성 설치
pip install fastapi uvicorn google-genai opensearch-py sentence-transformers kiwipiepy

# 환경 변수 설정
export GEMINI_API_KEY="..."
export APP_PASSWORD="..."        # 미설정 시 인증 없이 동작
export ADMIN_PASSWORD="..."      # 어드민 대시보드 접근용

# 서버 시작
cd chatbot
uvicorn app.py:app --reload
```

### 주요 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/auth` | 로그인 (username + password → token) |
| `GET` | `/api/conversations` | 내 대화 목록 |
| `POST` | `/api/conversations` | 새 대화 생성 |
| `POST` | `/api/chat` | 피드백 요청 (RAG 포함) |
| `DELETE` | `/api/conversations/{id}` | 대화 삭제 |
| `POST` | `/api/admin/auth` | 어드민 로그인 |
| `GET` | `/api/admin/users` | 전체 사용자 목록 |
| `GET` | `/api/admin/conversations` | 전체 대화 조회 |

### 화면

- `/` — 학습자용 쓰기 피드백 챗봇
- `/admin` — 어드민 대시보드 (사용자별 대화·턴 조회)

---

## 개발 환경

- Python 3.12 / venv: `.venv/`
- OpenSearch: `localhost:9200` (SSL 없음)
- 주요 패키지: `fastapi`, `google-genai`, `opensearch-py`, `sentence-transformers`, `kiwipiepy`, `rapidfuzz`, `pandas`
