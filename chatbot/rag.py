"""
RAG 모듈 — OpenSearch 하이브리드 검색 + 오류 주석 자연어 포매팅.

가설 4.ipynb (Cell 3~4) 에서 추출.
app.py 에서는 get_rag_examples() 만 호출한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 초기화 (모듈 로드 시 1회)
# ---------------------------------------------------------------------------
INDEX_NAME = "korean_test_0301"

_opensearch_client = None
_embed_model = None

try:
    from opensearchpy import OpenSearch

    _opensearch_client = OpenSearch(
        hosts=[{"host": "172.30.1.81", "port": 9200}],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )
    _opensearch_client.info()  # 연결 확인
    count = _opensearch_client.count(index=INDEX_NAME)["count"]
    print(f"[RAG] OpenSearch 연결 완료 — 인덱스 '{INDEX_NAME}' 문서 수: {count:,}")
except Exception as e:
    print(f"[RAG] OpenSearch 연결 실패 — RAG 비활성: {e}")
    _opensearch_client = None

try:
    from sentence_transformers import SentenceTransformer

    _model_path = str(Path(__file__).resolve().parent.parent.parent / "model" / "KURE-v1")
    _embed_model = SentenceTransformer(_model_path, local_files_only=True)
    print(f"[RAG] 임베딩 모델 로드 완료 (dim={_embed_model.get_sentence_embedding_dimension()})")
except Exception as e:
    print(f"[RAG] 임베딩 모델 로드 실패 — RAG 비활성: {e}")
    _embed_model = None

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
POS_NAMES = {
    "NNG": "명사", "NNP": "고유명사", "NNB": "의존명사", "NR": "수사", "NP": "대명사",
    "VV": "동사", "VA": "형용사", "VX": "보조용언",
    "VCP": "긍정 지정사", "VCN": "부정 지정사",
    "MM": "관형사", "MAG": "부사", "MAJ": "접속 부사", "IC": "감탄사",
    "JKS": "주격 조사", "JKC": "보격 조사", "JKG": "관형격 조사",
    "JKO": "목적격 조사", "JKB": "부사격 조사", "JKV": "호격 조사",
    "JKQ": "인용격 조사", "JX": "보조사", "JC": "접속 조사",
    "EP": "선어말 어미", "EF": "종결 어미", "EC": "연결 어미",
    "ETN": "명사형 전성 어미", "ETM": "관형형 전성 어미",
    "XPN": "체언 접두사", "XSN": "명사 파생 접미사", "XSV": "동사 파생 접미사",
    "XSA": "형용사 파생 접미사", "XSM": "부사 파생 접미사", "XR": "어근",
    "SF": "종결 부호", "SP": "구분 부호", "SS": "인용 부호",
    "SSO": "여는 부호", "SSC": "닫는 부호", "SE": "줄임표",
    "SO": "붙임표", "SW": "특수 문자", "SL": "알파벳", "SH": "한자",
    "SN": "숫자", "SB": "글머리",
    "UN": "분석 불능", "NONE": "분석 불능", "SYMBOL": "기호",
}

# ---------------------------------------------------------------------------
# 내부 함수
# ---------------------------------------------------------------------------

def _parse_morph_pos(token: str):
    """형태소/품사 문자열을 (형태소, 품사이름|None) 으로 분리."""
    parts = token.rsplit("/", 1)
    if len(parts) == 2:
        return parts[0], POS_NAMES.get(parts[1], parts[1])
    return token, None


def _extract_error_type(signature: str | None) -> str | None:
    """시그니처에서 오류 양상(REP/MIF/OM/ADD)을 추출."""
    if not signature:
        return None
    parts = signature.split(":")
    if len(parts) >= 2:
        return parts[1]
    return None


# ---------------------------------------------------------------------------
# 포매팅
# ---------------------------------------------------------------------------

def format_correction_pair(pair_str: str, signature: str | None = None, error_word: str | None = None) -> str:
    """교정 쌍 문자열을 자연어로 변환."""
    try:
        left, right = pair_str.split("→")
    except ValueError:
        return pair_str

    error_type = _extract_error_type(signature)
    ew = error_word if error_word else None

    # OM (누락)
    if left == "∅":
        corr_morph, corr_pos_name = _parse_morph_pos(right)
        base = f"'{corr_morph}' 빠짐 ({corr_pos_name} 누락)" if corr_pos_name else f"'{corr_morph}' 빠짐"
        if ew:
            return f"'{ew}' 뒤에 {base}"
        return base

    orig_morph, orig_pos_name = _parse_morph_pos(left)

    # ADD (첨가)
    if right == "∅" or right.startswith("ADD"):
        base = f"'{orig_morph}' 불필요 ({orig_pos_name} 삭제 필요)" if orig_pos_name else f"'{orig_morph}' 불필요"
        if ew:
            return f"'{ew}'에서 {base}"
        return base

    corr_morph, corr_pos_name = _parse_morph_pos(right)

    # 동일 형태소 (활용 오류)
    if orig_morph == corr_morph and orig_pos_name == corr_pos_name and orig_pos_name:
        base = f"'{orig_morph}' 활용 오류 ({orig_pos_name})"
        if ew:
            return f"'{ew}'에서 {base}"
        return base

    # 일반 케이스 (REP/MIF)
    if orig_pos_name and corr_pos_name:
        if orig_pos_name == corr_pos_name:
            if error_type == "MIF":
                base = f"'{orig_morph}' → '{corr_morph}' ({orig_pos_name} 철자 수정)"
            elif error_type == "REP":
                base = f"'{orig_morph}' → '{corr_morph}' ({orig_pos_name} 대치)"
            else:
                base = f"'{orig_morph}' → '{corr_morph}' ({orig_pos_name} 수정)"
        else:
            base = f"'{orig_morph}' → '{corr_morph}' ({orig_pos_name} → {corr_pos_name})"
    else:
        pos_desc = orig_pos_name or corr_pos_name
        if pos_desc:
            base = f"'{orig_morph}' → '{corr_morph}' ({pos_desc})"
        else:
            base = f"'{orig_morph}' → '{corr_morph}'"

    if ew:
        return f"'{ew}'에서 {base}"
    return base


def _format_hit(hit: dict) -> str:
    """검색 결과 한 건을 자연어 블록으로 변환."""
    lines = [f'문장: "{hit["original_text"]}"']
    sigs = hit.get("error_signatures", [])
    pairs = hit.get("correction_pairs", [])
    words = hit.get("error_words", [])
    for i, pair_str in enumerate(pairs):
        sig = sigs[i] if i < len(sigs) else None
        ew = words[i] if i < len(words) else None
        lines.append(f"교정: {format_correction_pair(pair_str, sig, ew)}")
    return "\n".join(lines)


def _format_rag_examples(hits: list[dict], max_examples: int = 3) -> str:
    """상위 N개 히트를 포매팅하여 하나의 문자열로 결합."""
    if not hits:
        return ""
    blocks = []
    for i, hit in enumerate(hits[:max_examples]):
        formatted = _format_hit(hit)
        formatted_lines = formatted.split("\n")
        numbered = f"{i + 1}. {formatted_lines[0]}"
        for line in formatted_lines[1:]:
            numbered += f"\n   {line}"
        blocks.append(numbered)
    return "[유사 오류 사례]\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 검색
# ---------------------------------------------------------------------------

def _search_hybrid(query_text: str, k: int = 10) -> list[dict]:
    """BM25 + 벡터 검색 후 결과 병합. 오류 문장만 반환."""
    _source = ["original_text", "error_signatures", "correction_pairs", "error_words", "has_error"]

    bm25_resp = _opensearch_client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"match": {"original_text": query_text}},
            "_source": _source,
        },
    )

    query_vec = _embed_model.encode(query_text).tolist()
    knn_resp = _opensearch_client.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": query_vec, "k": k}}},
            "_source": _source,
        },
    )

    merged = {}
    for hit in bm25_resp["hits"]["hits"] + knn_resp["hits"]["hits"]:
        merged[hit["_id"]] = hit["_source"]

    results = []
    for doc_id, src in merged.items():
        if src.get("has_error") is True:
            results.append({
                "original_text": src["original_text"],
                "error_signatures": src.get("error_signatures", []),
                "correction_pairs": src.get("correction_pairs", []),
                "error_words": src.get("error_words", []),
            })
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_rag_examples(text: str, k: int = 10, max_examples: int = 3) -> str:
    """학습자 문장으로 유사 오류 사례를 검색·포매팅하여 반환.

    RAG 구성 요소가 비활성이거나 검색 중 오류 발생 시 빈 문자열을 반환한다.
    """
    if not _opensearch_client or not _embed_model:
        print("[RAG] 검색 건너뜀 — OpenSearch 또는 임베딩 모델 미초기화")
        return ""
    try:
        hits = _search_hybrid(text, k=k)
        print(f"[RAG] 검색 완료 — 오류 문장 {len(hits)}건 / 포매팅 {min(len(hits), max_examples)}건")
        return _format_rag_examples(hits, max_examples=max_examples)
    except Exception as e:
        print(f"[RAG] 검색 실패: {e}")
        return ""
