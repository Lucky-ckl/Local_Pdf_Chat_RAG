"""
重排序器 —— 对检索结果进行二次精排

学习要点：
- 两阶段检索（Recall + Rerank）是工业界常用的范式
- Recall 阶段用高效检索（FAISS/BM25）从大量文档中召回候选
- Rerank 阶段用更精确的模型（交叉编码器/LLM）对候选精排
- 交叉编码器比双塔模型更精确，但速度更慢（适合对少量候选精排）
"""

import logging
import re
import threading
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from config import OLLAMA_MODEL_NAME, RERANK_METHOD, RERANK_MODEL_NAME

# 重排服务可用性状态（P1.1）：最近一次重排序是否因交叉编码器不可用而回退。
# 供调用方（core/retriever.py）读取并显式提示「本次结果未精排」；不区分用户主动关闭（P1.3 再区分）。
# score_source 取值（WBS §11.3 / P1.3）：结果是否真正经过交叉编码器重排。
SCORE_SOURCE_RERANK = "rerank"
SCORE_SOURCE_RRF = "rrf"


class RerankStatus(NamedTuple):
    """单次重排的状态契约（字段与 WBS §11.3 一致）。

    - score_source:            "rerank"（当前结果确实经过重排）或 "rrf"（未重排，结果来自融合排序）
    - rerank_disabled_by_user: 用户通过 RERANK_METHOD=none 主动关闭重排
    - rerank_unavailable:      重排服务故障，本次结果已回退到融合排序
    """

    score_source: str
    rerank_disabled_by_user: bool
    rerank_unavailable: bool


class RerankResult(list):
    """重排结果列表，附带本次请求的状态（RerankStatus）。

    继承 list 以保持既有返回结构兼容（调用方可继续按普通列表遍历/切片/取长度）；
    状态随返回值一并传递，属请求级状态，避免用共享可变全局判断单次请求结果（P1.3）。
    """

    def __init__(self, results, status):
        super().__init__(results)
        self.status = status


# 重排服务可用性状态（P1.1）：最近一次重排尝试是否因交叉编码器不可用而回退。
# 仅为向后兼容保留的「最近一次」镜像；单次请求的结果判断请使用 rerank_results(...).status，
# 不要读取该全局，以免并发请求之间相互污染（P1.3）。
rerank_unavailable = False

# 交叉编码器（懒加载 + 线程安全）
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """懒加载交叉编码器模型（双重检查锁定，线程安全）

    模型名来自 config.RERANK_MODEL_NAME（P1.2），不再硬编码。
    """
    global _cross_encoder, rerank_unavailable
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
                    logging.info("交叉编码器加载成功: %s", RERANK_MODEL_NAME)
                except Exception as e:
                    _cross_encoder = None
                    rerank_unavailable = True
                    logging.error(
                        "交叉编码器加载失败（模型=%s，异常类型=%s）：%s",
                        RERANK_MODEL_NAME, type(e).__name__, e,
                    )
    return _cross_encoder


def _normalize_cross_encoder_scores(raw_scores, expected_count):
    """校验并归一化交叉编码器输出分数。

    交叉编码器应为每个 (query, doc) 对返回一个实数分数。这里统一压平为一维 float
    列表，并校验：分数数量与候选文档数一致、且均为有限值（非 NaN/Inf）。
    返回顺序与输入顺序一一对应；异常时抛 ValueError，由调用方捕获并降级，
    避免形状/数值异常导致排序错乱或静默丢文档。
    """
    scores = np.asarray(raw_scores, dtype=float).reshape(-1)
    if scores.shape[0] != expected_count:
        raise ValueError(
            f"重排序分数数量与候选文档数量不一致：scores={scores.shape[0]}, docs={expected_count}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("重排序分数包含非有限值（NaN/Inf）")
    return [float(score) for score in scores]


def _rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k):
    """执行一次交叉编码器重排，返回 (results, unavailable)。

    unavailable=True 表示模型加载或预测失败、本次未真正重排（已按原序回退）。
    该标志通过返回值传递，不依赖模块级全局，避免并发请求互相污染。
    """
    global rerank_unavailable
    if not docs:
        return [], False

    encoder = get_cross_encoder()
    if encoder is None:
        rerank_unavailable = True
        logging.warning(
            "重排服务不可用（模型=%s）：本次结果未精排，已按原序回退", RERANK_MODEL_NAME
        )
        return _fallback_results(doc_ids, docs, metadata_list), True

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = _normalize_cross_encoder_scores(encoder.predict(cross_inputs), len(docs))
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': score})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k], False
    except Exception as e:
        rerank_unavailable = True
        logging.error(
            "交叉编码器重排序失败（模型=%s，异常类型=%s）：%s",
            RERANK_MODEL_NAME, type(e).__name__, e,
        )
        return _fallback_results(doc_ids, docs, metadata_list), True


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """使用交叉编码器对检索结果进行重排序（保留的公开接口，仅返回结果列表）。"""
    results, _ = _rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    return results


@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """使用 LLM 对查询和文档的相关性进行评分（带缓存）"""
    from utils.network import get_session
    try:
        prompt = f"""给定以下查询和文档片段，评估它们的相关性。
        评分标准：0分表示完全不相关，10分表示高度相关。
        只需返回一个0-10之间的整数分数，不要有任何其他解释。

        查询: {query}
        文档片段: {doc}
        相关性分数(0-10):"""

        response = get_session().post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180
        )
        result = response.json().get("response", "").strip()
        try:
            return max(0, min(10, float(result)))
        except ValueError:
            match = re.search(r'\b([0-9]|10)\b', result)
            return float(match.group(1)) if match else 5.0
    except Exception as e:
        logging.error(f"LLM评分失败: {str(e)}")
        return 5.0


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """使用 LLM 逐一评分进行重排序"""
    if not docs:
        return []
    results = []
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        score = get_llm_relevance_score(query, doc)
        results.append((doc_id, {'content': doc, 'metadata': meta, 'score': score / 10.0}))
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    return results[:top_k]


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """对检索结果进行重排序（统一入口）。

    返回值仍是结果列表（保持既有结构兼容），但为 RerankResult，附带本次请求的状态：
        result.status.score_source            —— "rerank" | "rrf"
        result.status.rerank_disabled_by_user —— 用户以 RERANK_METHOD=none 主动关闭
        result.status.rerank_unavailable      —— 重排服务故障导致回退
    调用方据此区分三种来源，不得把融合结果冒充为重排结果（WBS P1.3）。
    """
    global rerank_unavailable
    # 复位模块级镜像，使其只反映本次尝试（仅用于向后兼容的外部观察，不参与请求内判断）。
    rerank_unavailable = False
    if method is None:
        method = RERANK_METHOD

    if method == "none":
        # 用户主动关闭重排（配置级入口）：属正常行为，不计为服务故障、不产生告警。
        results = _fallback_results(doc_ids, docs, metadata_list)
        return RerankResult(results, RerankStatus(SCORE_SOURCE_RRF, True, False))

    if method == "llm":
        results = rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
        return RerankResult(results, RerankStatus(SCORE_SOURCE_RERANK, False, False))

    if method == "cross_encoder":
        results, unavailable = _rerank_with_cross_encoder(
            query, docs, doc_ids, metadata_list, top_k
        )
        rerank_unavailable = unavailable
        status = RerankStatus(
            SCORE_SOURCE_RRF if unavailable else SCORE_SOURCE_RERANK, False, unavailable
        )
        return RerankResult(results, status)

    # 未识别的取值：不执行重排。既非用户显式关闭，也不能静默——按服务不可用降级并告警。
    logging.warning("未识别的 RERANK_METHOD=%r，未执行重排，按服务不可用降级", method)
    rerank_unavailable = True
    results = _fallback_results(doc_ids, docs, metadata_list)
    return RerankResult(results, RerankStatus(SCORE_SOURCE_RRF, False, True))


def _fallback_results(doc_ids, docs, metadata_list):
    """回退方案：按原始顺序返回"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
