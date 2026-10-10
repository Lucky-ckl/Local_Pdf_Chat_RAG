"""
检索器 —— 混合检索 + 递归检索策略

学习要点：
- 混合检索（Hybrid Search）结合语义检索和关键词检索的优势
- alpha 参数控制两者权重（0.7 = 70% 语义 + 30% 关键词）
- 递归检索通过多轮迭代，利用 LLM 改写查询获取更全面的信息
"""

import logging
import math
import time
from dataclasses import dataclass
from config import HYBRID_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_K, MAX_RETRIEVAL_ITERATIONS
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.embeddings import encode_query
from core.reranker import rerank_results
from core import reranker
from features.web_search import check_serpapi_key, search_web


def _normalize_dense_similarities(distances):
    """把 FAISS 的真实距离映射为 [0,1] 语义相关性分数（越大越相关）。

    输入：FAISS L2 系列索引（IndexFlatL2 / IVFFlat / IVFPQ）返回的**平方 L2 距离**
    （``d >= 0``，越小代表越相似）。

    公式（可复现）：

        sim(d) = 1 / (1 + max(d, 0))

    - 输出范围 (0, 1]：``d = 0`` 时为 1.0，``d`` 越大越趋近 0；结果恒为有限值。
    - 单调递减：距离越小，语义相关性分数不会更低。
    - 非有限值（NaN / ±Inf）视为「最不相似」，记 0.0；负距离（不应出现）按下限 0 处理。

    方法选择：未采用 min-max 归一化 ``(d_max - d) / (d_max - d_min)``。min-max 依赖当次候选集合
    的最小/最大值，跨查询不可比，且单个离群点会压缩其余候选的分数；``1 / (1 + d)`` 是与候选集合
    无关的固定映射，跨查询可比、对候选集合变化稳定，也不要求向量已归一化。

    局限：对小距离区间分辨率偏低（相关文档分数集中于 1 附近），故 ``HYBRID_ALPHA`` 默认值需在 P5
    用评测校准，本阶段不做调整。
    """
    similarities = []
    for value in distances:
        try:
            distance = float(value)
        except (TypeError, ValueError):
            similarities.append(0.0)
            continue
        if not math.isfinite(distance):
            similarities.append(0.0)
            continue
        if distance < 0:
            distance = 0.0
        similarities.append(1.0 / (1.0 + distance))
    return similarities


def _extract_dense_similarities(semantic_results, expected_count):
    """从 ``semantic_results['distances']`` 读取真实距离并归一化为 [0,1]。

    缺失、结构异常或长度与候选数量不匹配时返回 ``None``；调用方据此把语义贡献按 0 处理并记录
    告警——绝不回退到基于排名序位的伪分数。
    """
    distances = semantic_results.get('distances')
    if not isinstance(distances, list) or len(distances) == 0:
        logging.warning("语义检索未提供真实距离（distances），语义贡献按 0 处理（不使用伪分数）")
        return None
    first = distances[0]
    try:
        length = len(first)
    except TypeError:
        logging.warning("语义距离结构异常，语义贡献按 0 处理")
        return None
    if length != expected_count:
        logging.warning(
            "语义距离数量(%s)与候选数量(%s)不匹配，语义贡献按 0 处理", length, expected_count
        )
        return None
    return _normalize_dense_similarities(first)


def hybrid_merge(semantic_results, bm25_results, alpha=None):
    """
    合并语义检索和 BM25 检索结果

    使用加权分数：语义相关性分数 × alpha + BM25分数 × (1-alpha)

    语义相关性分数来自 FAISS 真实距离的归一化结果（见 ``_normalize_dense_similarities``），
    不再使用基于排名序位的伪分数；两路分数均落在 [0,1]，故融合分数也在 [0,1]。

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]],
                           'distances': [[...]]}  # distances 为 FAISS 平方 L2 距离
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: 语义检索权重

    Returns:
        排序后的 [(doc_id, {'score': ..., 'content': ..., 'metadata': ...})]
    """
    if alpha is None:
        alpha = HYBRID_ALPHA

    merged_dict = {}

    # 处理语义检索结果
    if (semantic_results and
            isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
            isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
            isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
            isinstance(semantic_results['documents'][0], list) and
            len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(
                semantic_results['ids'][0])):
        ids = semantic_results['ids'][0]
        similarities = _extract_dense_similarities(semantic_results, len(ids))
        for i, (doc_id, doc, meta) in enumerate(
                zip(ids, semantic_results['documents'][0], semantic_results['metadatas'][0])):
            semantic_score = similarities[i] if similarities is not None else 0.0
            merged_dict[doc_id] = {
                'score': alpha * semantic_score, 'content': doc, 'metadata': meta,
                'dense_score': semantic_score, 'bm25_score': 0.0,
            }
    else:
        logging.warning("语义检索结果为空或格式异常")

    # 处理 BM25 结果（BM25 分数无上界，按其当次最大值归一到 [0,1]，越大越相关）
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)

    valid_scores = []
    for r in bm25_results:
        if isinstance(r, dict) and isinstance(r.get('score'), (int, float)):
            score = float(r['score'])
            if math.isfinite(score):
                valid_scores.append(score)
    max_bm25 = max(valid_scores) if valid_scores else 0.0

    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            continue
        doc_id = result['id']
        raw_score = result['score']
        if isinstance(raw_score, (int, float)) and math.isfinite(float(raw_score)) and max_bm25 > 0:
            norm_score = max(float(raw_score), 0.0) / max_bm25
        else:
            norm_score = 0.0

        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * norm_score
            merged_dict[doc_id]['bm25_score'] = norm_score
        else:
            metadata = vector_store.metadatas_map.get(doc_id, {})
            merged_dict[doc_id] = {
                'score': (1 - alpha) * norm_score,
                'content': result['content'], 'metadata': metadata,
                'dense_score': 0.0, 'bm25_score': norm_score,
            }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)


def reciprocal_rank_fusion(rankings, k=60):
    """倒数排名融合（Reciprocal Rank Fusion, RRF）。

    将多个「已按相关性降序排列」的 doc_id 列表融合为单一分数：
        score(d) = Σ_ranking 1 / (k + rank_ranking(d))     # rank 从 1 开始
    仅出现在部分检索源中的文档照常累加其所在排名的贡献。返回 {doc_id: rrf_score}。
    参考：Cormack et al. (2009), Reciprocal Rank Fusion。k 取 60 为业界常用经验值。
    """
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _rrf_fallback_ranking(hybrid, dense_ids, bm25_results, top_k):
    """重排不可用或被关闭时的降级结果：对 dense 与 BM25 两个排序执行真实 RRF 融合。

    hybrid 为 hybrid_merge 的融合候选 [(doc_id, data)]，提供 content/metadata 载荷。
    返回 [(doc_id, {'content', 'metadata', 'score'})]，按 RRF 分数降序取前 top_k。
    排序依据是 RRF 融合结果，而非未经融合的候选顺序，也不会伪造分数。
    """
    payload_by_id = {doc_id: data for doc_id, data in hybrid}
    bm25_ids = [r['id'] for r in (bm25_results or [])
                if isinstance(r, dict) and 'id' in r]
    rrf_scores = reciprocal_rank_fusion([list(dense_ids), bm25_ids])
    ordered_ids = sorted(
        payload_by_id,
        key=lambda doc_id: rrf_scores.get(doc_id, 0.0),
        reverse=True,
    )
    return [
        (doc_id, {
            'content': payload_by_id[doc_id]['content'],
            'metadata': payload_by_id[doc_id]['metadata'],
            'score': rrf_scores.get(doc_id, 0.0),
        })
        for doc_id in ordered_ids[:top_k]
    ]


@dataclass
class RewriteResult:
    """查询改写结果（WBS v1.0 P3.2）：供 P5 归因。"""

    rewritten: str
    rewrite_used: bool
    rewrite_rule: str


def _history_turns(history):
    """把 history 规整为 ``[(question, answer), ...]``；兼容 Turn 对象与 dict。"""
    turns = []
    for item in history or []:
        if isinstance(item, dict):
            question = item.get("question", "")
            answer = item.get("answer", "")
        else:
            question = getattr(item, "question", "")
            answer = getattr(item, "answer", "")
        if question or answer:
            turns.append((question, answer))
    return turns


def _default_rewrite_fn(prompt, model_choice):
    """默认改写实现（P3.2）：调用 LLM；返回错误占位串或空结果时视为**改写失败**并抛出，
    由 ``rewrite_query_with_history`` 回退原 query，避免把「未配置 Key / 服务错误」的提示串当成检索查询。"""
    from core.generator import call_llm_simple, is_llm_failure
    result = call_llm_simple(prompt, model_choice)
    if is_llm_failure(result):
        raise RuntimeError("查询改写未获得有效 LLM 结果（可能未配置 Key 或服务不可用）")
    return result


def rewrite_query_with_history(question, history=None, model_choice="siliconflow", rewrite_fn=None):
    """有历史时把「指代 / 省略」追问改写为自包含问题（WBS v1.0 P3.2）。

    - **无历史轮次 → 不改写**（``rewrite_used=False``），避免无谓的 LLM 调用与语义漂移；
    - 改写失败 / 超时 / 结果异常（空或过长）→ 回退原 query，不阻塞检索；
    - 返回 ``RewriteResult(rewritten, rewrite_used, rewrite_rule)`` 供 P5 归因。
    """
    turns = _history_turns(history)
    if not turns:
        return RewriteResult(question, False, "")

    conversation = "\n".join(f"用户：{q}\n助手：{a}" for q, a in turns[-3:])
    prompt = (
        "你是查询改写助手。请结合对话历史，把用户的最新问题改写为一个**自包含**的检索查询："
        "补全其中的指代（如「它 / 那台」）与省略的主语，使其脱离上下文也能独立理解。\n"
        "要求：\n"
        "1. 只输出改写后的查询，不要解释、不要加引号；\n"
        "2. 若最新问题本身已自包含，原样返回该问题；\n"
        "3. 不要引入历史中未出现的新信息。\n\n"
        f"[对话历史]\n{conversation}\n\n[最新问题]\n{question}\n\n[改写结果]"
    )
    try:
        fn = rewrite_fn or _default_rewrite_fn
        raw = fn(prompt, model_choice)
        rewritten = (raw or "").strip().strip('"').strip()
    except Exception as e:
        logging.warning("查询改写失败，回退原 query：%s", e)
        return RewriteResult(question, False, "")

    if not rewritten or len(rewritten) > 200:
        logging.warning("查询改写结果为空或过长（%d 字符），回退原 query", len(rewritten or ""))
        return RewriteResult(question, False, "")
    if rewritten == question:
        return RewriteResult(question, False, "")
    return RewriteResult(rewritten, True, "history_coreference")


def recursive_retrieval(initial_query, max_iterations=None, enable_web_search=False,
                        model_choice="siliconflow", history=None, rewrite_out=None, trace_out=None):
    """
    递归检索与查询优化

    流程：0.有历史时先做查询改写（P3.2）→ 1.语义+BM25检索 → 2.混合排序 → 3.重排序 →
          4.LLM判断是否改写query继续

    Args:
        history: 会话历史（list[Turn] 或 list[dict]）；仅非空时触发查询改写（P3.2）。
        rewrite_out: 可选 dict；写入 ``rewritten`` / ``rewrite_used`` / ``rewrite_rule``。
        trace_out: 可选 dict；写入本次召回候选的分数明细与 ``score_source``（供 SSE retrieval 事件）。

    Returns:
        (all_contexts, all_doc_ids, all_metadata)   # 保持既有三元组返回，兼容原有调用方
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    _rewrite_started = time.perf_counter()
    rewrite = rewrite_query_with_history(initial_query, history, model_choice)
    rewrite_ms = (time.perf_counter() - _rewrite_started) * 1000.0
    rewrite_fields = {
        "rewritten": rewrite.rewritten,
        "rewrite_used": rewrite.rewrite_used,
        "rewrite_rule": rewrite.rewrite_rule,
    }
    if rewrite_out is not None:
        rewrite_out.update(rewrite_fields)
    if trace_out is not None:
        trace_out.update(rewrite_fields)

    query = rewrite.rewritten
    all_contexts, all_doc_ids, all_metadata = [], [], []
    seen_web_sources = set()
    # 本次请求最后一轮的重排状态（请求级，避免读取模块级全局导致并发串扰）；None 表示本轮未执行重排。
    last_status = None
    # 轻量检索 trace（P3.5 / §11.3）：只记录**实际执行**的阶段与耗时，未执行的步骤不写入。
    trace = {
        "iterations": 0,
        "max_iterations": max_iterations,
        "rewrite_used": bool(rewrite.rewrite_used),
        "enable_web_search": bool(enable_web_search),
        "web_search_used": False,
        "stages_ms": {"rewrite": round(rewrite_ms, 3)},
    }

    def _add_stage_ms(name, elapsed_ms):
        trace["stages_ms"][name] = round(trace["stages_ms"].get(name, 0.0) + elapsed_ms, 3)

    for i in range(max_iterations):
        trace["iterations"] += 1
        logging.info(f"递归检索 {i + 1}/{max_iterations}，当前 Query: {query}")

        # 网络搜索补充
        web_texts = []
        if enable_web_search and check_serpapi_key():
            trace["web_search_used"] = True
            try:
                for res in search_web(query):
                    title = res.get('title') or ''
                    url = res.get('url') or ''
                    snippet = res.get('snippet') or ''
                    web_texts.append(f"标题：{title}\n摘要：{snippet}")
                    source_key = url or f"{title}\n{snippet}"
                    if snippet and source_key not in seen_web_sources:
                        seen_web_sources.add(source_key)
                        all_contexts.append(snippet)
                        all_doc_ids.append(f"web:{source_key}")
                        all_metadata.append({
                            'source': 'web',
                            'title': title,
                            'url': url,
                            'timestamp': res.get('timestamp'),
                        })
            except Exception as e:
                logging.error(f"网络搜索出错: {str(e)}")

        # 语义检索（连同真实平方 L2 距离，用于混合分数归一化）
        _stage = time.perf_counter()
        query_embedding = encode_query(query)
        sem_docs, sem_ids, sem_metas, sem_distances = vector_store.search_with_scores(
            query_embedding, k=RETRIEVAL_TOP_K
        )
        _add_stage_ms("semantic", (time.perf_counter() - _stage) * 1000.0)

        prepared = {
            "ids": [sem_ids],
            "documents": [sem_docs],
            "metadatas": [sem_metas],
            "distances": [sem_distances],
        }

        # BM25 检索
        _stage = time.perf_counter()
        bm25_res = bm25_manager.search(query, top_k=RETRIEVAL_TOP_K) if bm25_manager.bm25_index else []
        _add_stage_ms("bm25", (time.perf_counter() - _stage) * 1000.0)

        # 混合排序 → 重排序
        _stage = time.perf_counter()
        hybrid = hybrid_merge(prepared, bm25_res)
        _add_stage_ms("hybrid_merge", (time.perf_counter() - _stage) * 1000.0)
        ids_iter, docs_iter, meta_iter = [], [], []
        for doc_id, data in hybrid[:RETRIEVAL_TOP_K]:
            ids_iter.append(doc_id)
            docs_iter.append(data['content'])
            meta_iter.append(data['metadata'])

        if docs_iter:
            _stage = time.perf_counter()
            reranked = None
            try:
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
            except Exception as e:
                logging.error(f"重排序失败: {str(e)}")

            # 读取本次请求的状态（随返回值传递），避免读取模块级全局造成并发请求串扰。
            status = getattr(reranked, "status", None)
            if reranked is None:
                # 重排调用本身抛出异常：降级到真实 RRF 融合，并明确标记服务不可用。
                logging.warning(
                    "重排服务调用失败：本轮结果未经过重排序精排（score_source=rrf，已回退到 RRF 融合排序）"
                )
                reranked = _rrf_fallback_ranking(hybrid, sem_ids, bm25_res, RERANK_TOP_K)
                last_status = reranker.RerankStatus(reranker.SCORE_SOURCE_RRF, False, True)
            elif status is not None and status.score_source == reranker.SCORE_SOURCE_RRF:
                # 重排未生效（用户关闭或服务故障）：用真实 RRF 融合结果取代未经重排的候选顺序。
                reranked = _rrf_fallback_ranking(hybrid, sem_ids, bm25_res, RERANK_TOP_K)
                last_status = status
                if status.rerank_unavailable:
                    logging.warning(
                        "重排服务不可用：本轮结果未经过重排序精排（score_source=rrf，已回退到 RRF 融合排序）"
                    )
                elif status.rerank_disabled_by_user:
                    logging.info(
                        "重排已由用户关闭（RERANK_METHOD=none）：本轮使用 RRF 融合排序（score_source=rrf）"
                    )
            else:
                # 重排成功；status 为 None（旧式纯列表返回）时按重排成功处理。
                last_status = status or reranker.RerankStatus(reranker.SCORE_SOURCE_RERANK, False, False)
            _add_stage_ms("rerank", (time.perf_counter() - _stage) * 1000.0)
        else:
            reranked = []

        # 整合结果
        current_contexts = web_texts[:]
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                all_contexts.append(data['content'])
                all_metadata.append(data['metadata'])
            current_contexts.append(data['content'])

        if trace_out is not None:
            # 记录本轮回召候选的分数明细（供 P3.5 / P4.5 retrieval 事件；index 与数组下标一致）。
            score_source = last_status.score_source if last_status else reranker.SCORE_SOURCE_RRF
            payload_by_id = {doc_id: data for doc_id, data in hybrid}
            rank_before = {doc_id: pos for pos, (doc_id, _data) in enumerate(hybrid, start=1)}
            rank_after = {doc_id: pos for pos, (doc_id, _data) in enumerate(reranked, start=1)}
            chunks = []
            for pos, (doc_id, data) in enumerate(reranked[:RETRIEVAL_TOP_K]):
                metadata = data.get('metadata') or {}
                fused_payload = payload_by_id.get(doc_id) or {}
                chunks.append({
                    "index": pos,
                    "chunk_id": doc_id,
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page", 0),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "preview": (data.get('content') or "")[:200],
                    "dense_score": fused_payload.get('dense_score'),
                    "bm25_score": fused_payload.get('bm25_score'),
                    "fused_score": fused_payload.get('score'),
                    "rerank_score": data.get('score') if score_source == reranker.SCORE_SOURCE_RERANK else None,
                    "rank_before": rank_before.get(doc_id),
                    "rank_after": rank_after.get(doc_id),
                    "ocr_used": bool(metadata.get("ocr_used", False)),
                })
            trace_out["chunks"] = chunks
            trace_out["score_source"] = score_source
            trace_out["rerank_disabled_by_user"] = (
                bool(last_status.rerank_disabled_by_user) if last_status else False
            )
            trace_out["rerank_unavailable"] = bool(last_status.rerank_unavailable) if last_status else False
            trace_out["retrieval_trace"] = dict(trace)

        if i == max_iterations - 1:
            break

        # LLM 判断是否需要继续
        if current_contexts:
            summary = "\n".join(current_contexts[:3])
            prompt = f"""你是一个查询优化助手。根据以下信息判断是否需要新的查询。

[初始问题]
{initial_query}

[检索结果摘要]
{summary}

要求：
1. 如果信息已足够，直接回复：不需要进一步查询
2. 否则返回一个更精准的新查询，仅包含查询词
"""
            try:
                from core.generator import call_llm_simple
                next_query = call_llm_simple(prompt, model_choice)
                if "不需要" in next_query:
                    logging.info("LLM 判断无需更多查询")
                    break
                if len(next_query) > 100:
                    logging.warning("生成内容过长，不视为有效查询")
                    break
                query = next_query
                logging.info(f"生成下一轮查询: {query}")
            except Exception as e:
                logging.error(f"生成新查询失败: {str(e)}")
                break
        else:
            break

    return all_contexts, all_doc_ids, all_metadata
