"""P3.4 / P3.5 / P3.6 单测：事件顺序与字段、引用校验、检索/生成失败、心跳与断连。"""

import asyncio
import json
import logging
import re
import time

import pytest

import api_router
import core.generator as generator
from core.generator import stream_answer_events, verify_citations


class _DummyStore:
    is_ready = True


def _fake_retrieval_recorder(chunks, score_source="rerank", rewritten="rw", rewrite_used=True,
                             rerank_disabled_by_user=False, rerank_unavailable=False):
    def fake(initial_query, max_iterations=None, enable_web_search=False,
             model_choice="siliconflow", history=None, rewrite_out=None, trace_out=None):
        if trace_out is not None:
            trace_out.update({
                "chunks": chunks,
                "score_source": score_source,
                "rerank_disabled_by_user": rerank_disabled_by_user,
                "rerank_unavailable": rerank_unavailable,
                "rewritten": rewritten,
                "rewrite_used": rewrite_used,
                "rewrite_rule": "history_coreference" if rewrite_used else "",
                "retrieval_trace": {"iterations": 1, "stages_ms": {"semantic": 1.0}},
            })
        return ["context chunk"], ["d0"], [{"source": "a.pdf", "page": 1}]
    return fake


def _sample_chunk():
    return {
        "index": 0, "chunk_id": "d0", "source": "a.pdf", "page": 1, "chunk_index": 3, "preview": "ctx",
        "dense_score": 0.9, "bm25_score": 0.2, "fused_score": 0.7, "rerank_score": 0.8,
        "rank_before": 1, "rank_after": 1, "ocr_used": False,
    }


def test_event_order_and_fields(monkeypatch):
    monkeypatch.setattr(generator, "vector_store", _DummyStore())
    monkeypatch.setattr(generator, "recursive_retrieval", _fake_retrieval_recorder([_sample_chunk()]))

    def fake_stream(question, model_choice="siliconflow", prompt=None, **kwargs):
        yield "液压泵的"
        yield "保修期是 12 个月 [0]。"

    monkeypatch.setattr(generator, "stream_answer", fake_stream)

    events = list(stream_answer_events("那它的保修期呢", session_id="s1",
                                       history=[{"question": "q", "answer": "a"}]))
    names = [e["event"] for e in events]

    assert names[0] == "retrieval"
    assert names[-1] == "done"
    assert names[-2] == "citation"
    assert "delta" in names
    assert names.index("retrieval") < names.index("delta")

    retrieval = events[0]["data"]
    assert retrieval["chunks"][0]["index"] == 0
    assert retrieval["score_source"] == "rerank"
    assert retrieval["rewrite_used"] is True
    # §11.3 / P3.5：retrieval 载荷须区分用户关闭与服务故障，并带检索 trace。
    assert retrieval["rerank_disabled_by_user"] is False
    assert retrieval["rerank_unavailable"] is False
    assert retrieval["retrieval_trace"]["iterations"] == 1

    deltas = "".join(e["data"]["text"] for e in events if e["event"] == "delta")
    assert deltas == "液压泵的保修期是 12 个月 [0]。"
    assert events[-2]["data"]["citation_passed"] is True
    assert events[-1]["data"]["session_id"] == "s1"
    assert isinstance(events[-1]["data"]["latency_ms"], int)


def test_retrieval_failure_emits_error_without_delta(monkeypatch):
    monkeypatch.setattr(generator, "vector_store", _DummyStore())

    def boom(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(generator, "recursive_retrieval", boom)

    events = list(stream_answer_events("q"))

    assert [e["event"] for e in events] == ["error"]
    assert events[0]["data"]["code"] == "retrieval_failed"


def test_generation_failure_keeps_partial_and_appends_error(monkeypatch):
    monkeypatch.setattr(generator, "vector_store", _DummyStore())
    monkeypatch.setattr(generator, "recursive_retrieval", _fake_retrieval_recorder([_sample_chunk()]))

    def failing_stream(question, model_choice="siliconflow", prompt=None, **kwargs):
        yield "已经输出的正文"
        raise RuntimeError("stream broke")

    monkeypatch.setattr(generator, "stream_answer", failing_stream)

    events = list(stream_answer_events("q"))
    names = [e["event"] for e in events]

    assert names == ["retrieval", "delta", "error"]
    assert events[1]["data"]["text"] == "已经输出的正文"  # 已输出正文保留
    assert events[2]["data"]["code"] == "generation_failed"


def test_knowledge_base_empty_emits_error(monkeypatch):
    class NotReady:
        is_ready = False

    monkeypatch.setattr(generator, "vector_store", NotReady())
    events = list(stream_answer_events("q"))
    assert [e["event"] for e in events] == ["error"]
    assert events[0]["data"]["code"] == "knowledge_base_empty"


def test_verify_citations_returns_resolved_citations_and_flags_out_of_range():
    chunks = [_sample_chunk()]
    ok = verify_citations("见 [0]。", chunks)
    assert ok["citation_passed"] is True
    assert ok["unsupported_claims"] == []
    assert ok["citations"] == [{"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 3}]

    result = verify_citations("见 [0] 与 [3]。", chunks)
    assert result["citation_passed"] is False
    assert result["unsupported_claims"] == [3]
    # 只回填存在且可定位的引用（越界的 [3] 不进入 citations）。
    assert result["citations"] == [{"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 3}]


def test_verify_citations_without_reference_returns_empty_citations():
    chunks = [_sample_chunk()]
    assert verify_citations("无引用", chunks) == {
        "citations": [], "citation_passed": True, "unsupported_claims": []
    }


def test_verify_citations_multiple_references_keep_index_mapping():
    chunks = [
        {"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 0},
        {"index": 1, "source": "b.pdf", "page": 2, "chunk_index": 5},
    ]
    result = verify_citations("见 [1] 与 [0]。", chunks)
    # citations[].index 即候选在 chunks 数组中的下标，按 index 升序回填。
    assert result["citations"] == [
        {"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 0},
        {"index": 1, "source": "b.pdf", "page": 2, "chunk_index": 5},
    ]
    assert result["citation_passed"] is True


def test_verify_citations_missing_metadata_is_not_locatable():
    # 缺 source/page/chunk_index：不可靠定位，不回填、也不虚构 ""/0 默认值。
    result = verify_citations("[0]", [{"index": 0}])
    assert result == {"citations": [], "citation_passed": False, "unsupported_claims": [0]}


def test_verify_citations_partial_metadata_is_not_locatable():
    # 只缺 chunk_index 一项也视为不可定位（按「键是否存在」判定，而非只看是否有值）。
    result = verify_citations("[0]", [{"index": 0, "source": "a.pdf", "page": 1}])
    assert result["citations"] == []
    assert result["citation_passed"] is False
    assert result["unsupported_claims"] == [0]


def test_verify_citations_zero_page_and_chunk_index_are_valid():
    # page=0（非 PDF 约定）与 chunk_index=0（首块）是合法值，不得误判为元数据缺失。
    result = verify_citations(
        "[0]", [{"index": 0, "source": "note.txt", "page": 0, "chunk_index": 0}]
    )
    assert result["citations"] == [{"index": 0, "source": "note.txt", "page": 0, "chunk_index": 0}]
    assert result["citation_passed"] is True
    assert result["unsupported_claims"] == []


def test_verify_citations_mixed_locatable_and_missing_metadata():
    chunks = [
        {"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 2},
        {"index": 1, "source": "b.pdf"},  # 缺 page/chunk_index
    ]
    result = verify_citations("见 [0] 与 [1]。", chunks)
    assert result["citations"] == [{"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 2}]
    assert result["citation_passed"] is False
    assert result["unsupported_claims"] == [1]


def test_verify_citations_duplicate_reference_is_deduplicated():
    chunks = [{"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 0}]
    result = verify_citations("[0] ... [0]", chunks)
    assert result["citations"] == [{"index": 0, "source": "a.pdf", "page": 1, "chunk_index": 0}]
    assert result["citation_passed"] is True
    assert result["unsupported_claims"] == []


def test_verify_citations_empty_chunks_flags_reference():
    assert verify_citations("见 [0]", []) == {
        "citations": [], "citation_passed": False, "unsupported_claims": [0]
    }


def test_retrieval_event_surfaces_rerank_disable_and_failure(monkeypatch):
    monkeypatch.setattr(generator, "vector_store", _DummyStore())
    monkeypatch.setattr(
        generator, "recursive_retrieval",
        _fake_retrieval_recorder(
            [_sample_chunk()], score_source="rrf",
            rerank_disabled_by_user=True, rerank_unavailable=False,
        ),
    )
    monkeypatch.setattr(generator, "stream_answer", lambda *a, **k: iter(["正文"]))
    events = list(stream_answer_events("q"))
    retrieval = events[0]["data"]
    assert retrieval["score_source"] == "rrf"
    assert retrieval["rerank_disabled_by_user"] is True
    assert retrieval["rerank_unavailable"] is False


# ── P3.4 / P3.6：SSE 包装层（心跳 / 断连） ──────────────────────────────
class _FakeRequest:
    def __init__(self, disconnect_after=None):
        self.calls = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self):
        self.calls += 1
        return self.disconnect_after is not None and self.calls > self.disconnect_after


class _TrackingGenerator:
    def __init__(self, events):
        self._iterator = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self.closed = True


def _events():
    return [
        {"event": "retrieval", "data": {"chunks": [_sample_chunk()], "score_source": "rerank",
                                        "rewritten": "q", "rewrite_used": False, "rewrite_rule": ""}},
        {"event": "delta", "data": {"text": "hello "}},
        {"event": "delta", "data": {"text": "world [0]"}},
        {"event": "citation", "data": {"citation_passed": True, "unsupported_claims": []}},
        {"event": "done", "data": {"session_id": "s1", "latency_ms": 5, "token_usage": None}},
    ]


async def _collect(agen):
    return [frame async for frame in agen]


def test_sse_stream_formats_frames_and_records_session(monkeypatch):
    api_router.session_store.clear()
    monkeypatch.setattr(api_router, "stream_answer_events", lambda *a, **k: _TrackingGenerator(_events()))

    request = _FakeRequest(disconnect_after=None)
    payload = api_router.AskStreamRequest(question="q", session_id="sess-1")
    frames = asyncio.run(_collect(api_router._sse_event_stream(request, payload)))

    assert frames[0].startswith("event: retrieval\n")
    assert "event: delta" in frames[1]
    assert frames[-1].startswith("event: done\n")
    assert re.search(r"event: retrieval\ndata: .*\"score_source\": \"rerank\"", frames[0])

    history = api_router.session_store.history("sess-1")
    assert len(history) == 1
    assert history[0].answer == "hello world [0]"
    assert history[0].citations == [_sample_chunk()]  # [0] 对应候选下标 0


def test_sse_disconnect_closes_generator_and_logs(monkeypatch, caplog):
    api_router.session_store.clear()
    tracking = _TrackingGenerator(_events())
    monkeypatch.setattr(api_router, "stream_answer_events", lambda *a, **k: tracking)

    request = _FakeRequest(disconnect_after=0)  # 下发首帧后即视为断连
    payload = api_router.AskStreamRequest(question="q", session_id="sess-2")

    with caplog.at_level(logging.INFO, logger="rag-api"):
        frames = asyncio.run(_collect(api_router._sse_event_stream(request, payload)))

    assert tracking.closed is True  # 断连后终止底层生成器（终止 LLM 调用）
    assert len(frames) == 1  # 断连后不再继续下发
    assert any("客户端断连" in record.getMessage() for record in caplog.records)
    assert api_router.session_store.history("sess-2") == []  # 未正常完成，不写入会话


def test_sse_emits_heartbeat_when_generation_stalls(monkeypatch):
    monkeypatch.setattr(api_router, "SSE_HEARTBEAT_SEC", 0.05)
    api_router.session_store.clear()

    class _SlowGenerator:
        def __init__(self):
            self._step = 0

        def __next__(self):
            self._step += 1
            if self._step == 1:
                return _events()[0]
            if self._step == 2:
                time.sleep(0.2)  # 制造超过心跳间隔的停顿
                return _events()[-1]
            raise StopIteration

        def __iter__(self):
            return self

        def close(self):
            pass

    monkeypatch.setattr(api_router, "stream_answer_events", lambda *a, **k: _SlowGenerator())
    request = _FakeRequest(disconnect_after=None)
    payload = api_router.AskStreamRequest(question="q")
    frames = asyncio.run(_collect(api_router._sse_event_stream(request, payload)))

    assert any(frame.startswith(": heartbeat") for frame in frames)
