"""P5.2 检索指标函数单元测试（WBS v1.0 §8 P5.2）。

覆盖 `eval/run_eval.py::_metrics_for` 的 question 级指标：
Hit@k / MRR@k / Recall@k，以及空结果、重复结果、k 边界等情形。

说明：只测试指标计算逻辑，不修改评测实现，不加载模型、不联网。
"""

import pytest

from eval.run_eval import KS, _metrics_for


def _row(question_id, flags, field="relevant"):
    """构造一行评测记录；flags 为各排名的相关性（True = 相关）。"""
    return {
        "question_id": question_id,
        "ranked": [{field: bool(f)} for f in flags],
    }


def _metrics(rows, counts, key="relevant", with_recall=True):
    """按默认口径调用被测量的指标函数。"""
    return _metrics_for(rows, counts, key=key, with_recall=with_recall)


def test_ks_is_3_5_10():
    assert KS == [3, 5, 10]


def test_hit_and_mrr_when_relevant_first():
    metrics = _metrics([_row("q1", [True, False, False])], {"q1": 1})

    assert metrics["n_questions"] == 1
    for k in KS:
        assert metrics[f"Hit@{k}"] == 1.0
        assert metrics[f"MRR@{k}"] == 1.0


def test_mrr_uses_reciprocal_rank_of_first_hit():
    metrics = _metrics([_row("q1", [False, False, True])], {"q1": 1})

    assert metrics["MRR@3"] == round(1 / 3, 4)
    assert metrics["MRR@5"] == round(1 / 3, 4)


def test_hit_at_k_boundary_positions():
    # 相关项恰好落在第 4 位：不在 k=3 内，但在 k=5 / k=10 内。
    metrics = _metrics([_row("q1", [False, False, False, True])], {"q1": 1})

    assert metrics["Hit@3"] == 0.0
    assert metrics["MRR@3"] == 0.0
    assert metrics["Hit@5"] == 1.0
    assert metrics["MRR@5"] == 0.25
    assert metrics["Hit@10"] == 1.0


def test_hit_at_k_exactly_at_boundary():
    # 相关项正好在第 k 位：k=3 应命中，排名倒数 = 1/3。
    metrics = _metrics([_row("q1", [False, False, True])], {"q1": 1})

    assert metrics["Hit@3"] == 1.0
    assert metrics["MRR@3"] == round(1 / 3, 4)


def test_relevant_beyond_max_k_is_missed():
    # 相关项在第 11 位：超出最大的 k=10，三档 Hit 均为 0。
    metrics = _metrics([_row("q1", [False] * 10 + [True])], {"q1": 1})

    for k in KS:
        assert metrics[f"Hit@{k}"] == 0.0
        assert metrics[f"MRR@{k}"] == 0.0


def test_results_are_truncated_at_k():
    # 第 10、11 位各有一个相关片段；k=10 只计前 10 个（1 个）。
    metrics = _metrics([_row("q1", [False] * 9 + [True, True])], {"q1": 2})

    assert metrics["Hit@10"] == 1.0
    assert metrics["Recall@10"] == 0.5


def test_fewer_results_than_k_does_not_error():
    metrics = _metrics([_row("q1", [False, True])], {"q1": 1})

    assert metrics["Hit@3"] == 1.0
    assert metrics["Hit@10"] == 1.0
    assert metrics["MRR@10"] == 0.5


def test_recall_uses_relevant_counts_as_denominator():
    rows = [_row("q1", [True, True, False])]

    assert _metrics(rows, {"q1": 2})["Recall@3"] == 1.0
    assert _metrics(rows, {"q1": 4})["Recall@3"] == 0.5


def test_recall_is_capped_at_one():
    # 命中的相关片段数多于标注的相关总数时，Recall 封顶为 1.0。
    metrics = _metrics([_row("q1", [True, True, True])], {"q1": 1})

    assert metrics["Recall@3"] == 1.0


def test_missing_relevant_count_defaults_to_one():
    rows = [_row("qX", [False, True])]

    assert _metrics(rows, {})["Recall@3"] == 1.0


def test_empty_retrieval_returns_zeros():
    metrics = _metrics([_row("q1", [])], {"q1": 1})

    for k in KS:
        assert metrics[f"Hit@{k}"] == 0.0
        assert metrics[f"MRR@{k}"] == 0.0
        assert metrics[f"Recall@{k}"] == 0.0


def test_no_questions_returns_zero_metrics():
    metrics = _metrics([], {})

    assert metrics["n_questions"] == 0
    for k in KS:
        assert metrics[f"Hit@{k}"] == 0.0
        assert metrics[f"MRR@{k}"] == 0.0


def test_duplicate_results_are_counted():
    # 同一相关片段重复出现三次：排名倒数取第一处，Recall 因封顶不虚高。
    metrics = _metrics([_row("q1", [True, True, True])], {"q1": 1})

    assert metrics["Hit@3"] == 1.0
    assert metrics["MRR@3"] == 1.0
    assert metrics["Recall@3"] == 1.0


def test_duplicates_do_not_exceed_relevant_total():
    # 重复命中不会让 Recall 超过「相关总数」所能支持的上限。
    metrics = _metrics([_row("q1", [True, True, True, True])], {"q1": 2})

    assert metrics["Recall@5"] == 1.0


def test_macro_average_over_questions():
    rows = [_row("q1", [True]), _row("q2", [False, False, False, False])]
    metrics = _metrics(rows, {"q1": 1, "q2": 1})

    assert metrics["n_questions"] == 2
    assert metrics["Hit@3"] == 0.5
    assert metrics["MRR@3"] == 0.5
    assert metrics["Recall@3"] == 0.5


def test_without_recall_omits_recall_keys():
    metrics = _metrics([_row("q1", [True])], {"q1": 1}, with_recall=False)

    assert "Hit@3" in metrics and "MRR@3" in metrics
    assert not any(name.startswith("Recall@") for name in metrics)


def test_custom_key_selects_other_relevance_field():
    rows = [{
        "question_id": "q1",
        "ranked": [{"relevant": False, "page_relevant": True}],
    }]

    assert _metrics(rows, {"q1": 1}, key="page_relevant")["Hit@3"] == 1.0
    assert _metrics(rows, {"q1": 1}, key="relevant")["Hit@3"] == 0.0


def test_abstain_style_rows_are_not_special_cased():
    # 应拒答题由调用方在传入前剔除；此处验证「无相关标注」时行为一致。
    metrics = _metrics([_row("q24", [False, False, False])], {"q24": 1})

    assert metrics["Hit@3"] == 0.0
    assert metrics["Recall@3"] == 0.0


def test_metrics_are_rounded_to_four_decimals():
    rows = [_row("q1", [False, False, True]), _row("q2", [False, False, False])]
    metrics = _metrics(rows, {"q1": 1, "q2": 1})

    assert metrics["MRR@3"] == round((1 / 3) / 2, 4)
    assert metrics["Hit@3"] == 0.5


def test_relevance_key_is_required():
    """相关性字段名必填，避免误用默认口径。"""
    with pytest.raises(TypeError):
        _metrics_for([_row("q1", [True])], {"q1": 1})
