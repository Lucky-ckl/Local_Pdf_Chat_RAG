"""P1.3 重排降级契约测试（WBS §P1.3 / §11.3）。

覆盖：
- 三种 score_source 来源可区分：rerank / rrf(用户关闭) / rrf(服务故障)；
- 用户 RERANK_METHOD=none 走 RRF 且不报「服务故障」；
- 服务故障（模型不可用 / predict 异常）回退到 RRF 并标记 rerank_unavailable；
- RRF 回退是真实融合排序，而非未经融合的候选顺序；
- 状态为请求级，不因其它请求的失败而被污染。

全部通过 monkeypatch 隔离：不加载真实模型、不发起网络请求。
"""

import logging
import threading

import numpy as np
import pytest

import core.reranker as reranker
import core.retriever as retriever


class _FakeEncoder:
    def __init__(self, scores=None, error=None):
        self._scores = scores
        self._error = error

    def predict(self, cross_inputs):
        if self._error is not None:
            raise self._error
        return self._scores


@pytest.fixture(autouse=True)
def _reset_rerank_state(monkeypatch):
    monkeypatch.setattr(reranker, "_cross_encoder", None)
    monkeypatch.setattr(reranker, "rerank_unavailable", False)


def _run(method="cross_encoder", top_k=2):
    return reranker.rerank_results(
        "q",
        ["doc-a", "doc-b"],
        ["id-a", "id-b"],
        [{"source": "a"}, {"source": "b"}],
        method=method,
        top_k=top_k,
    )


# ── 1. 三种来源可区分 ─────────────────────────────────────────────
def test_success_status_is_rerank(monkeypatch):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: _FakeEncoder(scores=[0.1, 0.9]))
    result = _run()
    assert result.status.score_source == "rerank"
    assert result.status.rerank_disabled_by_user is False
    assert result.status.rerank_unavailable is False


def test_user_disabled_status_is_rrf_disabled(monkeypatch):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: pytest.fail("不应加载模型"))
    result = _run(method="none")
    assert result.status.score_source == "rrf"
    assert result.status.rerank_disabled_by_user is True
    assert result.status.rerank_unavailable is False


def test_model_unavailable_status_is_rrf_unavailable(monkeypatch):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: None)
    result = _run()
    assert result.status.score_source == "rrf"
    assert result.status.rerank_disabled_by_user is False
    assert result.status.rerank_unavailable is True


def test_predict_error_status_is_rrf_unavailable(monkeypatch):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: _FakeEncoder(error=ValueError("boom")))
    result = _run()
    assert result.status.score_source == "rrf"
    assert result.status.rerank_unavailable is True


def test_user_disabled_and_service_failure_are_distinguishable(monkeypatch):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: None)
    failure = _run()
    disabled = _run(method="none")
    assert failure.status != disabled.status
    assert (failure.status.rerank_unavailable, failure.status.rerank_disabled_by_user) == (True, False)
    assert (disabled.status.rerank_unavailable, disabled.status.rerank_disabled_by_user) == (False, True)


# ── 2. 状态为请求级，不被并发请求污染 ───────────────────────────
def test_status_is_not_polluted_by_stale_global(monkeypatch):
    monkeypatch.setattr(reranker, "rerank_unavailable", True)
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: _FakeEncoder(scores=[0.9, 0.1]))
    result = _run()
    assert result.status.score_source == "rerank"
    assert result.status.rerank_unavailable is False


def test_concurrent_requests_keep_independent_status(monkeypatch):
    barrier = threading.Barrier(2)

    class _ConditionalEncoder:
        def predict(self, cross_inputs):
            if any(query == "BOOM" for query, _ in cross_inputs):
                raise RuntimeError("boom")
            return [0.5] * len(cross_inputs)

    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: _ConditionalEncoder())
    statuses = {}

    def _worker(query, key):
        barrier.wait()
        statuses[key] = reranker.rerank_results(
            query, ["doc"], ["id"], [{}], method="cross_encoder", top_k=1
        ).status

    threads = [
        threading.Thread(target=_worker, args=("ok", "ok")),
        threading.Thread(target=_worker, args=("BOOM", "boom")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert statuses["ok"] == reranker.RerankStatus("rerank", False, False)
    assert statuses["boom"] == reranker.RerankStatus("rrf", False, True)


# ── 3. 调用方（retriever）使用请求级状态，而非模块级全局 ────────
def _setup_single_doc(monkeypatch):
    monkeypatch.setattr(retriever, "encode_query", lambda q: np.zeros((1, 384), dtype="float32"))
    monkeypatch.setattr(
        retriever.vector_store,
        "search_with_scores",
        lambda query_embedding, k: (["ctx-A"], ["A"], [{"source": "A"}], [0.05]),
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)


def test_retriever_trusts_request_status_not_global(monkeypatch, caplog):
    _setup_single_doc(monkeypatch)

    def _fake_rerank(query, docs, ids, metadata, top_k):
        # 本次请求实际成功；但模块级全局残留了其它请求的失败痕迹。
        reranker.rerank_unavailable = True
        return reranker.RerankResult(
            [(doc_id, {"content": d, "metadata": m, "score": 1.0})
             for doc_id, d, m in zip(ids, docs, metadata)],
            reranker.RerankStatus(reranker.SCORE_SOURCE_RERANK, False, False),
        )

    monkeypatch.setattr(retriever, "rerank_results", _fake_rerank)
    with caplog.at_level(logging.WARNING):
        contexts, doc_ids, metadata = retriever.recursive_retrieval("q", max_iterations=1)

    assert contexts == ["ctx-A"]
    assert not any("重排服务不可用" in r.getMessage() for r in caplog.records)


def test_retriever_falls_back_to_rrf_on_service_failure(monkeypatch, caplog):
    _setup_single_doc(monkeypatch)
    monkeypatch.setattr(
        retriever,
        "rerank_results",
        lambda query, docs, ids, metadata, top_k: reranker.RerankResult(
            [], reranker.RerankStatus(reranker.SCORE_SOURCE_RRF, False, True)
        ),
    )
    with caplog.at_level(logging.WARNING):
        contexts, doc_ids, metadata = retriever.recursive_retrieval("q", max_iterations=1)

    assert doc_ids == ["A"]
    assert contexts == ["ctx-A"]
    assert any("重排服务不可用" in r.getMessage() for r in caplog.records)


def test_user_disabled_does_not_report_service_failure(monkeypatch, caplog):
    _setup_single_doc(monkeypatch)
    monkeypatch.setattr(reranker, "RERANK_METHOD", "none")
    with caplog.at_level(logging.INFO):
        contexts, doc_ids, metadata = retriever.recursive_retrieval("q", max_iterations=1)

    assert doc_ids == ["A"]
    assert not any("重排服务不可用" in r.getMessage() for r in caplog.records)
    assert any("用户关闭" in r.getMessage() for r in caplog.records)


# ── 4. 降级为真实 RRF 融合，而非候选顺序 ────────────────────────
def test_reciprocal_rank_fusion_scores():
    scores = retriever.reciprocal_rank_fusion([["A", "B"], ["B", "A"]], k=60)
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_fallback_is_real_fusion_not_candidate_order(monkeypatch):
    # dense 排序：B > C > A；BM25 排序：A > C。
    # RRF 融合：A(1/63+1/61) > C(1/62+1/62) > B(1/61) → [A, C, B]，与 dense/候选顺序 [B, C, A] 不同。
    monkeypatch.setattr(retriever, "encode_query", lambda q: np.zeros((1, 384), dtype="float32"))
    monkeypatch.setattr(
        retriever.vector_store,
        "search_with_scores",
        lambda query_embedding, k: (
            ["ctx-B", "ctx-C", "ctx-A"],
            ["B", "C", "A"],
            [{"source": "B"}, {"source": "C"}, {"source": "A"}],
            [0.10, 0.20, 0.30],
        ),
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", object())
    monkeypatch.setattr(
        retriever.bm25_manager,
        "search",
        lambda query, top_k: [
            {"id": "A", "score": 10.0, "content": "ctx-A"},
            {"id": "C", "score": 5.0, "content": "ctx-C"},
        ],
    )
    monkeypatch.setattr(reranker, "RERANK_METHOD", "none")

    contexts, doc_ids, metadata = retriever.recursive_retrieval("q", max_iterations=1)

    assert doc_ids == ["A", "C", "B"]
    assert contexts == ["ctx-A", "ctx-C", "ctx-B"]
