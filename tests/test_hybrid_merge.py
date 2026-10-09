"""P1.4 混合检索分数可信化测试（WBS §P1.4）。

覆盖：
- 语义分数来自归一化后的真实 FAISS 距离，而非基于排名序位的伪分数；
- 距离越小，语义相关性分数不会更低（单调）；
- 相同候选排名下，真实距离不同 → 语义分数不同；
- 语义分数与融合分数落在 [0,1] 且为有限值；
- 极端值（0 / 负 / 极大 / NaN / Inf）、空候选、缺失/错位距离的既定契约；
- vector_store.search 三元组契约保持；search_with_scores 的距离与候选一一对应且最近者最小。

全部离线：不加载模型、不访问网络。
"""

import logging
import math

import numpy as np
import pytest

import core.retriever as retriever
from core.retriever import _normalize_dense_similarities, hybrid_merge
from core.vector_store import VectorStore


def _semantic(ids, distances, documents=None, metadatas=None):
    documents = documents if documents is not None else [f"doc-{i}" for i in ids]
    metadatas = metadatas if metadatas is not None else [{"source": i} for i in ids]
    return {
        "ids": [list(ids)],
        "documents": [list(documents)],
        "metadatas": [list(metadatas)],
        "distances": [list(distances)],
    }


def _scores(merged):
    return {doc_id: data["score"] for doc_id, data in merged}


# ── 1. 语义分数来自真实距离 ───────────────────────────────────────
def test_closer_distance_yields_higher_score_even_when_ranked_later():
    # far 排在前面（索引 0）但距离更大；near 排在后面（索引 1）但距离更小。
    # 旧的序位伪分数会把 far 评得更高；真实距离归一化后 near 必须更高。
    merged = hybrid_merge(_semantic(["far", "near"], [9.0, 0.0]), [], alpha=0.7)
    scores = _scores(merged)
    assert scores["near"] > scores["far"]
    assert [doc_id for doc_id, _ in merged][0] == "near"


def test_same_rank_different_distance_changes_semantic_score():
    closer = hybrid_merge(_semantic(["x"], [0.1]), [], alpha=1.0)
    farther = hybrid_merge(_semantic(["x"], [5.0]), [], alpha=1.0)
    assert _scores(closer)["x"] > _scores(farther)["x"]


def test_normalized_similarity_is_monotonic_and_bounded():
    sims = _normalize_dense_similarities([0.0, 0.5, 1.0, 4.0, 100.0])
    assert sims == sorted(sims, reverse=True)
    assert all(0.0 <= s <= 1.0 and math.isfinite(s) for s in sims)
    assert sims[0] == pytest.approx(1.0)


# ── 2. 极端值 / 非有限值 / 空候选 / 缺失距离 ──────────────────────
def test_extreme_and_non_finite_distances_are_handled():
    sims = _normalize_dense_similarities(
        [0.0, -3.0, float("inf"), float("-inf"), float("nan")]
    )
    assert sims[0] == 1.0          # d = 0 → 最相似
    assert sims[1] == 1.0          # 负距离按下限 0 处理
    assert sims[2] == 0.0          # +Inf → 最不相似
    assert sims[3] == 0.0          # -Inf → 非有限，记 0
    assert sims[4] == 0.0          # NaN → 记 0
    assert all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in sims)


def test_empty_candidates_return_empty():
    assert hybrid_merge({"ids": [[]], "documents": [[]], "metadatas": [[]]}, []) == []
    assert hybrid_merge({}, []) == []


def test_missing_distances_contribute_zero_not_pseudo_score(caplog):
    caplog.set_level(logging.WARNING)
    semantic = {"ids": [["doc-a"]], "documents": [["a"]], "metadatas": [[{"source": "a"}]]}
    merged = hybrid_merge(semantic, [], alpha=0.7)
    assert _scores(merged)["doc-a"] == 0.0
    assert any("未提供真实距离" in r.getMessage() for r in caplog.records)


def test_misaligned_distances_treated_as_missing(caplog):
    caplog.set_level(logging.WARNING)
    semantic = _semantic(["a", "b"], [0.1])  # 距离长度与候选数量不匹配
    merged = hybrid_merge(semantic, [], alpha=1.0)
    assert set(_scores(merged)) == {"a", "b"}
    assert all(score == 0.0 for score in _scores(merged).values())
    assert any("不匹配" in r.getMessage() for r in caplog.records)


# ── 3. 融合与 BM25 ───────────────────────────────────────────────
def test_fused_scores_stay_within_unit_interval():
    merged = hybrid_merge(
        _semantic(["a", "b"], [0.0, 2.0]),
        [
            {"id": "b", "score": 3.0, "content": "doc-b"},
            {"id": "c", "score": 6.0, "content": "doc-c"},
        ],
        alpha=0.6,
    )
    assert set(_scores(merged)) == {"a", "b", "c"}
    assert all(0.0 <= data["score"] <= 1.0 and math.isfinite(data["score"]) for _, data in merged)


def test_bm25_larger_score_contributes_more():
    merged = hybrid_merge(
        _semantic(["a", "b"], [0.0, 0.0]),  # 语义分数相同
        [
            {"id": "a", "score": 1.0, "content": "a"},
            {"id": "b", "score": 9.0, "content": "b"},
        ],
        alpha=0.5,
    )
    scores = _scores(merged)
    assert scores["b"] > scores["a"]


def test_negative_and_non_finite_bm25_scores_are_sanitized():
    merged = hybrid_merge(
        _semantic(["a"], [0.0]),
        [
            {"id": "a", "score": -5.0, "content": "a"},
            {"id": "b", "score": float("nan"), "content": "b"},
            {"id": "c", "score": 4.0, "content": "c"},
        ],
        alpha=0.5,
    )
    assert all(0.0 <= data["score"] <= 1.0 and math.isfinite(data["score"]) for _, data in merged)


# ── 4. vector_store 距离与 search 契约 ───────────────────────────
def test_search_keeps_three_tuple_and_search_with_scores_returns_distances():
    store = VectorStore()
    vectors = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype="float32",
    )
    store.build_index(["x", "y", "z"], ["cx", "cy", "cz"], [{}, {}, {}], vectors)

    docs, doc_ids, metadatas = store.search(vectors[0:1], k=3)
    assert doc_ids[0] == "cx"
    assert isinstance(docs, list) and isinstance(metadatas, list)

    docs2, ids2, metas2, distances = store.search_with_scores(vectors[0:1], k=3)
    assert ids2 == doc_ids
    assert len(distances) == len(ids2) == 3
    assert distances[0] == pytest.approx(0.0, abs=1e-6)   # 与查询相同的向量距离为 0
    assert distances == sorted(distances)                 # L2 距离升序 → 最近者在前


def test_recursive_retrieval_uses_real_distances_for_ordering(monkeypatch):
    monkeypatch.setattr(retriever, "encode_query", lambda q: np.zeros((1, 384), dtype="float32"))

    def _fake_search_with_scores(query_embedding, k):
        # 列表顺序与距离不一致：far 在前但距离更大。
        return (
            ["ctx-far", "ctx-near"],
            ["far", "near"],
            [{"source": "far"}, {"source": "near"}],
            [9.0, 0.0],
        )

    monkeypatch.setattr(retriever.vector_store, "search_with_scores", _fake_search_with_scores)
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)
    monkeypatch.setattr(
        retriever,
        "rerank_results",
        lambda query, docs, ids, metadata, top_k: [
            (doc_id, {"content": doc, "metadata": meta, "score": 1.0})
            for doc_id, doc, meta in zip(ids, docs, metadata)
        ],
    )

    contexts, doc_ids, metadata = retriever.recursive_retrieval("q", max_iterations=1)

    # 距离更小的 near 应排在 far 之前，证明 retriever 把真实距离传递给了 hybrid_merge。
    assert doc_ids == ["near", "far"]
    assert contexts == ["ctx-near", "ctx-far"]
