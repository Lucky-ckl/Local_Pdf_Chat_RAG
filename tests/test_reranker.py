"""P1.1 重排失败可见化测试。

覆盖：
- 重排成功：按交叉编码器分数排序，rerank_unavailable 为 False；
- 重排不可用（模型加载失败）：置 rerank_unavailable=True、输出明确日志、按原序回退且不抛异常；
- 重排执行失败（predict 抛错）：置位并记录异常类型，仍按原序回退；
- 调用方（core.retriever）能读取该状态并显式告警，问答流程不中断。

所有用例都通过 monkeypatch 隔离，不加载真实模型、不发起网络请求。
"""

import logging

import numpy as np
import pytest

import core.reranker as reranker
import core.retriever as retriever


class _FakeEncoder:
    """最小可用的交叉编码器替身：predict 返回预设分数或抛出预设异常。"""

    def __init__(self, scores=None, error=None):
        self._scores = scores
        self._error = error

    def predict(self, cross_inputs):
        if self._error is not None:
            raise self._error
        return self._scores


@pytest.fixture(autouse=True)
def _reset_rerank_state(monkeypatch):
    """每个用例开始前复位重排状态与缓存，结束后由 monkeypatch 自动还原。"""
    monkeypatch.setattr(reranker, "_cross_encoder", None)
    monkeypatch.setattr(reranker, "rerank_unavailable", False)


def test_rerank_success_orders_by_cross_encoder_scores(monkeypatch):
    monkeypatch.setattr(
        reranker, "get_cross_encoder", lambda: _FakeEncoder(scores=[0.1, 0.9, 0.5])
    )

    results = reranker.rerank_results(
        "query",
        ["doc-a", "doc-b", "doc-c"],
        ["id-a", "id-b", "id-c"],
        [{"source": "a"}, {"source": "b"}, {"source": "c"}],
        method="cross_encoder",
        top_k=3,
    )

    assert [doc_id for doc_id, _ in results] == ["id-b", "id-c", "id-a"]
    assert results[0][1]["score"] == pytest.approx(0.9)
    assert reranker.rerank_unavailable is False


def test_rerank_unavailable_falls_back_visibly(monkeypatch, caplog):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: None)

    with caplog.at_level(logging.WARNING):
        results = reranker.rerank_results(
            "query",
            ["doc-a", "doc-b"],
            ["id-a", "id-b"],
            [{"source": "a"}, {"source": "b"}],
            method="cross_encoder",
            top_k=2,
        )

    assert [doc_id for doc_id, _ in results] == ["id-a", "id-b"]
    assert reranker.rerank_unavailable is True
    assert any("重排服务不可用" in record.getMessage() for record in caplog.records)


def test_rerank_load_failure_logs_model_and_exception_type(monkeypatch, caplog):
    class _BoomCrossEncoder:
        def __init__(self, model_name):
            raise RuntimeError(f"模型不存在: {model_name}")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _BoomCrossEncoder)
    monkeypatch.setattr(reranker, "RERANK_MODEL_NAME", "definitely/not-a-real-model")

    with caplog.at_level(logging.ERROR):
        encoder = reranker.get_cross_encoder()

    assert encoder is None
    assert reranker.rerank_unavailable is True
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "definitely/not-a-real-model" in logged
    assert "RuntimeError" in logged


def test_rerank_predict_failure_sets_flag_and_keeps_order(monkeypatch, caplog):
    monkeypatch.setattr(
        reranker,
        "get_cross_encoder",
        lambda: _FakeEncoder(error=ValueError("predict exploded")),
    )

    with caplog.at_level(logging.ERROR):
        results = reranker.rerank_results(
            "query",
            ["doc-a", "doc-b"],
            ["id-a", "id-b"],
            [{"source": "a"}, {"source": "b"}],
            method="cross_encoder",
            top_k=2,
        )

    assert [doc_id for doc_id, _ in results] == ["id-a", "id-b"]
    assert reranker.rerank_unavailable is True
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "ValueError" in logged


def test_recursive_retrieval_uses_rerank_state_and_does_not_crash(monkeypatch, caplog):
    monkeypatch.setattr(
        retriever, "encode_query", lambda query: np.zeros((1, 384), dtype="float32")
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "search_with_scores",
        lambda query_embedding, k: (
            ["local context"],
            ["doc-local"],
            [{"source": "local.pdf"}],
            [0.2],
        ),
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)
    monkeypatch.setattr(reranker, "RERANK_METHOD", "cross_encoder")

    def _boom(model_name):
        raise RuntimeError("cannot load reranker")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _boom)

    with caplog.at_level(logging.WARNING):
        contexts, doc_ids, metadata = retriever.recursive_retrieval(
            "test query", max_iterations=1
        )

    assert contexts == ["local context"]
    assert doc_ids == ["doc-local"]
    assert reranker.rerank_unavailable is True
    assert any("重排服务不可用" in record.getMessage() for record in caplog.records)


def test_configured_model_name_is_passed_to_cross_encoder(monkeypatch):
    captured = {}

    class _RecordingEncoder:
        def __init__(self, model_name):
            captured["model_name"] = model_name

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _RecordingEncoder)
    monkeypatch.setattr(reranker, "RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

    encoder = reranker.get_cross_encoder()

    assert isinstance(encoder, _RecordingEncoder)
    assert captured["model_name"] == "BAAI/bge-reranker-base"


def test_reranker_model_name_matches_config():
    import config

    assert reranker.RERANK_MODEL_NAME == config.RERANK_MODEL_NAME


def test_rerank_accepts_2d_score_array(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_cross_encoder",
        lambda: _FakeEncoder(scores=np.array([[0.2], [0.7], [0.4]])),
    )

    results = reranker.rerank_results(
        "query",
        ["doc-a", "doc-b", "doc-c"],
        ["id-a", "id-b", "id-c"],
        [{"source": "a"}, {"source": "b"}, {"source": "c"}],
        method="cross_encoder",
        top_k=3,
    )

    assert [doc_id for doc_id, _ in results] == ["id-b", "id-c", "id-a"]
    assert reranker.rerank_unavailable is False


def test_score_count_mismatch_falls_back_visibly(monkeypatch, caplog):
    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: _FakeEncoder(scores=[0.5]))

    with caplog.at_level(logging.ERROR):
        results = reranker.rerank_results(
            "query",
            ["doc-a", "doc-b"],
            ["id-a", "id-b"],
            [{"source": "a"}, {"source": "b"}],
            method="cross_encoder",
            top_k=2,
        )

    assert [doc_id for doc_id, _ in results] == ["id-a", "id-b"]
    assert reranker.rerank_unavailable is True
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "分数数量" in logged
    assert "ValueError" in logged
