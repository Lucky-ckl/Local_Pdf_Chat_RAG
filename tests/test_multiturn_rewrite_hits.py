"""P3.7 多轮省略追问端到端（3 组样本）：history → 改写 → 命中正确 chunk。

用真实 BM25 检索 + 确定性的改写函数（模拟 LLM 改写输出，避免依赖外部 LLM）。
"""

import numpy as np
import pytest

import core.retriever as retriever
from core.bm25_index import BM25IndexManager


CORPUS = {
    "hyd": "挖掘机液压泵的保修期为 12 个月。",
    "engine": "挖掘机发动机的保养周期为 500 小时。",
    "track": "推土机履带的价格为 8000 元。",
}


def _rewrite_oracle(prompt, model_choice):
    """确定性的改写 oracle：根据 prompt 里的「最新问题」给出自包含查询。"""
    if "保修期" in prompt:
        return "挖掘机液压泵的保修期"
    if "保养" in prompt or "多久" in prompt:
        return "挖掘机发动机的保养周期"
    if "多少钱" in prompt or "价格" in prompt:
        return "推土机履带的价格"
    return "未知查询"


@pytest.mark.parametrize(
    "history_question, followup, expected_doc",
    [
        ("这台挖掘机的液压泵", "那它的保修期呢", "hyd"),
        ("这台挖掘机的发动机", "它多久保养一次", "engine"),
        ("那台推土机的履带", "那个多少钱", "track"),
    ],
)
def test_followup_rewrite_hits_expected_chunk(monkeypatch, history_question, followup, expected_doc):
    manager = BM25IndexManager()
    manager.build_index(list(CORPUS.values()), list(CORPUS.keys()))
    monkeypatch.setattr(retriever, "bm25_manager", manager)

    # 语义通道置空，只用真实 BM25；重排用占位实现（不加载模型）。
    monkeypatch.setattr(retriever, "encode_query", lambda q: np.zeros((1, 384), dtype="float32"))
    monkeypatch.setattr(
        retriever.vector_store, "search_with_scores", lambda embedding, k: ([], [], [], [])
    )
    monkeypatch.setattr(
        retriever,
        "rerank_results",
        lambda query, docs, ids, metadata, top_k: [
            (doc_id, {"content": doc, "metadata": meta, "score": 1.0})
            for doc_id, doc, meta in zip(ids, docs, metadata)
        ],
    )
    monkeypatch.setattr(retriever, "_default_rewrite_fn", _rewrite_oracle)

    rewrite_out = {}
    history = [{"question": history_question, "answer": "（历史回答）"}]
    _, doc_ids, _ = retriever.recursive_retrieval(
        followup, max_iterations=1, history=history, rewrite_out=rewrite_out
    )

    assert rewrite_out["rewrite_used"] is True  # 省略追问被改写
    assert expected_doc in doc_ids  # 命中正确 chunk
