"""P3.2 查询改写单测：无历史不改写、有历史改写、失败回退、检索使用改写结果。"""

import numpy as np

import core.retriever as retriever
from core.retriever import rewrite_query_with_history


def test_no_history_does_not_rewrite():
    called = {"n": 0}

    def fake(prompt, model):
        called["n"] += 1
        return "should not be used"

    result = rewrite_query_with_history("那它的保修期呢", history=None, rewrite_fn=fake)

    assert result.rewrite_used is False
    assert result.rewritten == "那它的保修期呢"
    assert result.rewrite_rule == ""
    assert called["n"] == 0  # 无历史不得触发 LLM 改写


def test_history_triggers_rewrite_with_rule():
    seen = {}

    def fake(prompt, model):
        seen["prompt"] = prompt
        return "挖掘机液压泵的保修期是多久"

    result = rewrite_query_with_history(
        "那它的保修期呢",
        history=[{"question": "这台挖掘机的液压泵", "answer": "液压泵位于机身右侧。"}],
        rewrite_fn=fake,
    )

    assert result.rewrite_used is True
    assert result.rewritten == "挖掘机液压泵的保修期是多久"
    assert result.rewrite_rule == "history_coreference"
    assert "对话历史" in seen["prompt"]
    assert "这台挖掘机的液压泵" in seen["prompt"]


def test_rewrite_failure_falls_back_to_original():
    def boom(prompt, model):
        raise RuntimeError("timeout")

    result = rewrite_query_with_history(
        "那它呢", history=[{"question": "q", "answer": "a"}], rewrite_fn=boom
    )

    assert result.rewrite_used is False
    assert result.rewritten == "那它呢"


def test_empty_or_overlong_rewrite_falls_back():
    history = [{"question": "q", "answer": "a"}]
    blank = rewrite_query_with_history("q2", history=history, rewrite_fn=lambda p, m: "   ")
    assert blank.rewrite_used is False and blank.rewritten == "q2"

    overlong = rewrite_query_with_history("q2", history=history, rewrite_fn=lambda p, m: "x" * 300)
    assert overlong.rewrite_used is False and overlong.rewritten == "q2"


def test_unchanged_rewrite_is_not_marked_used():
    result = rewrite_query_with_history(
        "q2", history=[{"question": "q", "answer": "a"}], rewrite_fn=lambda p, m: "q2"
    )
    assert result.rewrite_used is False


def test_recursive_retrieval_uses_rewritten_query(monkeypatch):
    queries = []

    def fake_encode(query):
        queries.append(query)
        return np.zeros((1, 384), dtype="float32")

    monkeypatch.setattr(retriever, "encode_query", fake_encode)
    monkeypatch.setattr(
        retriever.vector_store, "search_with_scores", lambda embedding, k: ([], [], [], [])
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)
    monkeypatch.setattr(retriever, "_default_rewrite_fn", lambda prompt, model: "自包含的问题")

    rewrite_out = {}
    retriever.recursive_retrieval(
        "那它呢",
        max_iterations=1,
        history=[{"question": "这台挖掘机的液压泵", "answer": "..."}],
        rewrite_out=rewrite_out,
    )

    assert queries == ["自包含的问题"]
    assert rewrite_out["rewrite_used"] is True
    assert rewrite_out["rewritten"] == "自包含的问题"
    assert rewrite_out["rewrite_rule"] == "history_coreference"


def test_llm_error_placeholder_is_detected_as_failure():
    from core.generator import is_llm_failure

    assert is_llm_failure("错误：未配置 SiliconFlow API Key。") is True
    assert is_llm_failure("调用SiliconFlow API时出错: 401") is True
    assert is_llm_failure("") is True
    assert is_llm_failure(None) is True
    assert is_llm_failure("挖掘机液压泵的保修期") is False


def test_default_rewrite_falls_back_when_llm_returns_error_string(monkeypatch):
    # 未配置 Key 时 call_llm_simple 返回错误占位串，必须视为「改写失败」并回退原 query，
    # 不能把错误串当成检索查询（P3.2 / Gradio 多轮入口修复）。
    monkeypatch.setattr(
        "core.generator.call_llm_simple",
        lambda prompt, model: "错误：未配置 SiliconFlow API Key。",
    )
    result = rewrite_query_with_history(
        "那它的保修期呢", history=[{"question": "这台挖掘机的液压泵", "answer": "..."}]
    )
    assert result.rewrite_used is False
    assert result.rewritten == "那它的保修期呢"
