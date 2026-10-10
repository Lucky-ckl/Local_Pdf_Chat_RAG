"""P3.3 多轮上下文压缩单测：最近一轮完整、更早轮压缩、摘要缓存、长度受控。"""

import core.generator as generator
from core.generator import _build_prompt, compress_history


def test_compress_history_empty_returns_blank():
    assert compress_history([]) == ""
    assert compress_history(None) == ""


def test_compress_history_single_turn_keeps_full_qa():
    out = compress_history([{"question": "q1", "answer": "a1"}], cache={}, model_choice="x")
    assert "q1" in out and "a1" in out
    assert "摘要" not in out  # 单轮无更早轮次，不应有摘要块


def test_summary_generated_once_and_cached(monkeypatch):
    calls = {"n": 0}

    def fake_llm(prompt, model):
        calls["n"] += 1
        return "此前讨论了液压泵与保修"

    monkeypatch.setattr(generator, "call_llm_simple", fake_llm)
    cache = {}
    history = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
    ]

    first = compress_history(history, cache=cache, model_choice="x")
    second = compress_history(history, cache=cache, model_choice="x")

    assert calls["n"] == 1  # 同一批更早轮次只生成一次摘要
    assert first == second
    assert "此前讨论了液压泵与保修" in first
    assert "q3" in first and "a3" in first  # 最近一轮完整保留


def test_prompt_length_bounded_regardless_of_turn_count():
    history = [
        {"question": "q" * 100, "answer": "a" * 200} for _ in range(10)
    ]
    out = compress_history(history, cache={}, model_choice="x", max_chars=500)
    assert len(out) <= 500


def test_summary_failure_falls_back_to_bounded_truncation(monkeypatch):
    def boom(prompt, model):
        raise RuntimeError("llm down")

    monkeypatch.setattr(generator, "call_llm_simple", boom)
    out = compress_history(
        [{"question": "important question", "answer": "a"}, {"question": "q2", "answer": "a2"}],
        cache={},
        model_choice="x",
    )
    assert "important question" in out  # 降级为关键问句截断，仍保留信息


def test_build_prompt_includes_history_block():
    prompt = _build_prompt(
        "Q", "ctx", False, True, False, False,
        history_block="[上一轮问答]\n用户：q1\n助手：a1",
    )
    assert "对话历史" in prompt
    assert "q1" in prompt


def test_build_prompt_without_history_block_unchanged():
    prompt = _build_prompt("Q", "ctx", False, True, False, False)
    assert "对话历史" not in prompt
