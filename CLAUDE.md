# Korean Morpheme Error Analysis Project

## Overview
한국어 학습자 말뭉치에서 오류 문장을 검색/분석하는 연구 프로젝트. 입력 문장의 오류 유형(오타/문법)을 판별하고, 말뭉치 내 유사 오류 사례를 검색한다.

## Architecture
- **OpenSearch**: BM25(키워드) + kNN 벡터 검색 (하이브리드)
- **Embedding**: `KURE-v1` (1024차원, `sentence-transformers`)
- **형태소 분석**: `kiwipiepy` (입력 문장 토큰화)
- **말뭉치**: `말뭉치.parquet.gzip` (표본 번호, 문장, 형태 주석 컬럼)

## Key Concepts
- BM25 유사도 > 벡터 유사도: 오타 오류 가능성 높음
- 벡터 유사도 > BM25 유사도: 의미적 유사성만 존재
- 형태소 배열 비교에는 레벤슈타인 거리 사용 (`rapidfuzz`)
- 오류 유형: 오타 오류 / 문법 오류 / 혼합

## OpenSearch Index Schema (`korean_test`)
| Field | Type | Description |
|-------|------|-------------|
| `original_text` | text (nori analyzer) | 원본 문장 |
| `morphs` | keyword | 형태소 태그 배열 |
| `embedding` | knn_vector (1024d, HNSW cosine) | 문장 임베딩 |
| `error_signatures` | keyword[] | 오류 시그니처 (`오류위치:오류양상:오류층위`) |
| `correction_pairs` | keyword[] | 교정 쌍 (`원형태소/품사→교정형태소/품사`) |
| `error_patterns` | keyword[] | 오류 양상 (REP, MIF, OM, ADD) |
| `has_error` | boolean | 오류 존재 여부 |

- 전체 7,749건 중 오류 문장 3,725건, 정상 문장 3,728건 (고유 텍스트 7,453건)

## 말뭉치 오류 주석 컬럼
| 컬럼 | 설명 | 예시 |
|------|------|------|
| `오류 위치` | 오류 발생 형태소 위치 코드 | CNNG, FAP, FOP, FED, FNP |
| `오류 양상` | REP(대치), MIF(오형), OM(누락), ADD(첨가) | REP 66K, MIF 30K, OM 24K |
| `오류 층위` | PP(표기), MCJ(형태소결합), DS(방언), ST(문체) 등 | PP 53K, MCJ 6K |
| `교정 형태소`/`교정 주석` | 교정된 형태소와 품사 | 하고→와, JKB→JKB |

- 오류 주석이 있는 문장: 65,791개 / 전체 235,902개 (약 28%)
- 고유 오류 시그니처: 1,644종, 고유 교정 쌍: 28,357종

## Search Pipeline
1. 입력 문장 형태소 분석 -> 문장 + 형태소 배열 생성
2. OpenSearch에서 BM25 / 벡터 검색 각각 수행
3. 두 결과의 스코어 비교로 오류 유형 추정
4. 오류 시그니처/교정 쌍 기반 term 검색 (직접 검색 시 정밀도 높음)
5. 2단계 검색: BM25/벡터 후보 → 오류 패턴 역추정 → 시그니처 2차 검색

## Dev Environment
- Python 3.12, venv: `.venv/`
- OpenSearch: localhost:9200 (SSL 없음, 인증 없음)
- Dependencies: `pandas`, `sentence-transformers`, `opensearch-py`, `kiwipiepy`, `rapidfuzz`

## Files
- `가설 1.ipynb`: BM25/벡터/하이브리드 검색 비교 실험
- `가설 2.ipynb`: kiwipiepy 형태소 분석 및 말뭉치 품사태깅 비교, 가중치 N-gram 유사도
- `가설 3.ipynb`: 오류 주석 활용 검색 정확도 개선 실험
- `말뭉치.parquet.gzip`: 학습자 오류 말뭉치 데이터
- `model/KURE-v1/`: 한국어 임베딩 모델 (로컬)

## 실험 결과 요약

### 가설 1~2: 문장 수준 유사도 검색
- BM25 + 벡터 + 형태소 N-gram 조합으로 "유사 문장"은 찾지만 "유사 오류"는 잘 못 찾음

### 가설 3: 오류 주석 활용
- **3-1/3-2 (직접 시그니처/교정 쌍 검색)**: 오류 유형을 알면 정밀도 매우 높음
- **3-3 (2단계 검색)**: Stage 1(BM25/벡터) 후보의 오류 패턴을 역추정 → 노이즈에 취약
  - 오타/철자 오류: BM25가 오타 단어를 매칭 못함 → 후보 품질 낮음 → 패턴 추정 실패
  - 조사 오류: Stage 1에서 패턴 추정이 비교적 합리적
  - 핵심 병목: **입력 문장의 오류 유형을 모르는 상태에서 역추정하는 과정**

## TODO
- **LLM 기반 오류 진단 → 구조 검색**: LLM이 입력 오류를 진단 → 시그니처/교정 쌍 생성 → 3-1/3-2로 직접 검색 (가장 유망)
- MCP 활용 (LLM이 직접 오류 판단 후 말뭉치 능동 검색)
- 출신 국가 등 메타데이터 필터링
