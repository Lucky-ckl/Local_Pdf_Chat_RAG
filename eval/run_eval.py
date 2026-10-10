"""P5.2 检索指标脚本（WBS v1.0 §8 P5.2）。

只做**检索**，不跑生成（不调用任何 LLM）。同一份脚本同时用于：
- 当前代码库（改造后：②③④ 组）；
- `git archive 38b39f6` 导出的**改造前**代码库（① 组）。

指标：Hit@k / MRR@k / Recall@k（k=3/5/10），question 级计算后取**宏平均**（§1.1.7），
报告同时给出有效样本数与 k 值。缺标注的样本（应拒答题）不计入检索指标分母，
只单独统计。

用法（由调用方通过环境变量切换配置，保证配置在不同组之间可复现）：
    $env:RERANK_METHOD="cross_encoder"; $env:OCR_ENABLED="true"
    .\.venv\Scripts\python.exe eval\run_eval.py --group full --index-dir eval\index\full
"""

import argparse
import datetime
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KS = [3, 5, 10]


def _basename(value):
    return os.path.basename((value or "").replace("\\", "/"))


def _load_eval_set(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _discover_corpus(corpus_dir):
    exts = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}
    files = [p for p in sorted(pathlib.Path(corpus_dir).iterdir())
             if p.is_file() and p.suffix.lower() in exts]
    return files


def _ingest_modern(paths):
    """改造后：按页 Block → chunk 元数据（P2.2 / P2.7）。"""
    from core.document_loader import load_document
    from core.text_splitter import split_blocks

    chunks, metadatas, ids, per_file = [], [], [], []
    for path in paths:
        try:
            blocks = load_document(str(path))
            text = "\n".join(b.text for b in blocks)
            if not text.strip():
                raise ValueError("文档内容为空或无法提取文本")
            parts = split_blocks(blocks)
            for i, chunk in enumerate(parts):
                meta = {
                    "source": chunk.source or path.name,
                    "doc_id": f"{path.stem}_{i}",
                    "page": getattr(chunk, "page", None),
                    "chunk_index": getattr(chunk, "chunk_index", i),
                    "ocr_used": bool(getattr(chunk, "ocr_used", False)),
                }
                chunks.append(chunk.text)
                metadatas.append(meta)
                ids.append(f"{path.stem}_{i}")
            per_file.append({"file": path.name, "chunks": len(parts),
                             "ocr_used": any(bool(getattr(c, "ocr_used", False)) for c in parts)})
        except Exception as exc:
            per_file.append({"file": path.name, "chunks": 0, "error": str(exc)})
    return chunks, ids, metadatas, per_file


def _ingest_legacy(paths):
    """改造前：纯文本抽取（无页码 / 无 OCR）。"""
    from core.document_loader import extract_text
    from core.text_splitter import split_text

    chunks, metadatas, ids, per_file = [], [], [], []
    for path in paths:
        try:
            text = extract_text(str(path))
            if not text or not text.strip():
                raise ValueError("文档内容为空或无法提取文本")
            parts = split_text(text)
            for i, part in enumerate(parts):
                meta = {"source": path.name, "doc_id": f"{path.stem}_{i}", "page": None}
                chunks.append(part)
                metadatas.append(meta)
                ids.append(f"{path.stem}_{i}")
            per_file.append({"file": path.name, "chunks": len(parts), "ocr_used": False})
        except Exception as exc:
            per_file.append({"file": path.name, "chunks": 0, "error": str(exc)})
    return chunks, ids, metadatas, per_file


def _build_index(paths, index_dir):
    import core.document_loader as loader
    from core.bm25_index import bm25_manager
    from core.embeddings import encode_texts
    from core.vector_store import vector_store

    modern = hasattr(loader, "load_document")
    vector_store.clear()
    bm25_manager.clear()

    t0 = time.perf_counter()
    ingest_fn = _ingest_modern if modern else _ingest_legacy
    chunks, ids, metadatas, per_file = ingest_fn(paths)
    parse_ms = (time.perf_counter() - t0) * 1000.0

    if not chunks:
        raise SystemExit("语料入库后没有任何 chunk，无法评测")

    t0 = time.perf_counter()
    embeddings = encode_texts(chunks, show_progress=False)
    embed_ms = (time.perf_counter() - t0) * 1000.0

    vector_store.build_index(chunks, ids, metadatas, embeddings)
    bm25_manager.build_index(chunks, ids)

    persisted = None
    if hasattr(__import__("core.vector_store", fromlist=["x"]), "save_persisted_index"):
        from core.vector_store import save_persisted_index
        persisted = bool(save_persisted_index(index_dir=index_dir))

    docs = {}
    for meta in metadatas:
        entry = docs.setdefault(_basename(meta.get("source")),
                                {"source": _basename(meta.get("source")), "chunks": 0,
                                 "ocr_used": False, "pages": set()})
        entry["chunks"] += 1
        if meta.get("ocr_used"):
            entry["ocr_used"] = True
        if isinstance(meta.get("page"), int) and meta["page"] > 0:
            entry["pages"].add(meta["page"])
    doc_list = []
    for entry in docs.values():
        pages = sorted(entry.pop("pages"))
        entry["page_count"] = len(pages)
        doc_list.append(entry)
    doc_list.sort(key=lambda d: d["source"])

    chunk_keys = [[_basename(m.get("source")), m.get("page")] for m in metadatas]

    return {
        "modern": modern,
        "total_chunks": int(vector_store.total_chunks),
        "documents": doc_list,
        "per_file": per_file,
        "chunk_keys": chunk_keys,
        "timing_ms": {"parse_and_ocr": round(parse_ms, 3), "embed": round(embed_ms, 3)},
        "persisted": persisted,
    }


def _retrieve(question, history):
    from core.retriever import recursive_retrieval
    import inspect

    params = inspect.signature(recursive_retrieval).parameters
    kwargs = {"initial_query": question, "max_iterations": 1, "enable_web_search": False}
    trace = {}
    if "history" in params:
        kwargs["history"] = history or None
    if "trace_out" in params:
        kwargs["trace_out"] = trace
    contexts, doc_ids, metadatas = recursive_retrieval(**kwargs)
    return contexts, doc_ids, metadatas, trace


def _metrics_for(rows, relevant_counts, key, with_recall=True):
    out = {"n_questions": len(rows)}
    for k in KS:
        hits, rr, recalls = [], [], []
        for row in rows:
            ranked = row["ranked"][:k]
            first = None
            got = 0
            for pos, doc in enumerate(ranked, start=1):
                if doc[key]:
                    got += 1
                    if first is None:
                        first = pos
            hits.append(1.0 if first is not None else 0.0)
            rr.append(1.0 / first if first is not None else 0.0)
            total = relevant_counts.get(row["question_id"], 0) or 1
            recalls.append(min(got / total, 1.0))
        n = max(len(rows), 1)
        out[f"Hit@{k}"] = round(sum(hits) / n, 4)
        out[f"MRR@{k}"] = round(sum(rr) / n, 4)
        if with_recall:
            out[f"Recall@{k}"] = round(sum(recalls) / n, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--corpus", default="eval/corpus")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import config

    eval_set = _load_eval_set(args.eval_set)
    corpus = _discover_corpus(args.corpus)
    index_dir = args.index_dir
    if index_dir is None:
        index_dir = getattr(config, "INDEX_DIR", "data/index/")

    index_info = _build_index(corpus, index_dir)

    relevant_counts_page = {}
    relevant_counts_source = {}
    relevant_counts_kw = {}
    for it in eval_set:
        if it.get("should_abstain"):
            continue
        src = _basename(it.get("expected_source"))
        page = it.get("expected_page")
        relevant_counts_source[it["question_id"]] = sum(
            1 for s, _p in index_info["chunk_keys"] if s == src)
        relevant_counts_page[it["question_id"]] = sum(
            1 for s, p in index_info["chunk_keys"]
            if s == src and (page is None or p == page))
        # 答案关键词命中为二值判定：分母固定为 1，故 Recall 不再单列。
        relevant_counts_kw[it["question_id"]] = 1

    rows = []
    for it in eval_set:
        started = time.perf_counter()
        contexts, doc_ids, metadatas, trace = _retrieve(it["question"], it.get("history"))
        elapsed = (time.perf_counter() - started) * 1000.0
        expected = _basename(it.get("expected_source"))
        ranked = []
        for pos, (doc_id, text, meta) in enumerate(zip(doc_ids, contexts, metadatas), start=1):
            source = _basename((meta or {}).get("source"))
            page = (meta or {}).get("page")
            kws = it.get("expected_keywords") or []
            source_rel = (not it.get("should_abstain")) and source == expected
            page_rel = source_rel and (
                it.get("expected_page") is None or page == it.get("expected_page"))
            kw_hit = [w for w in kws if w and w in (text or "")]
            ranked.append({
                "rank": pos,
                "doc_id": doc_id,
                "source": source,
                "page": page,
                "relevant": source_rel,
                "source_relevant": source_rel,
                "page_relevant": page_rel,
                "kw_relevant": source_rel and bool(kw_hit),
                "keyword_hit": kw_hit,
                "snippet": (text or "")[:160],
            })
            if pos >= 10:
                break
        rows.append({
            "question_id": it["question_id"],
            "type": it["type"],
            "question": it["question"],
            "should_abstain": bool(it.get("should_abstain")),
            "ocr_required": bool(it.get("ocr_required")),
            "expected_source": it.get("expected_source"),
            "expected_page": it.get("expected_page"),
            "rewrite_used": bool(trace.get("rewrite_used")),
            "stages_ms": (trace.get("retrieval_trace") or {}).get("stages_ms"),
            "retrieval_ms": round(elapsed, 3),
            "ranked": ranked,
        })

    scored = [r for r in rows if not r["should_abstain"]]
    subsets = {
        "all": scored,
        "scanned": [r for r in scored if r["ocr_required"]],
        "non_scanned": [r for r in scored if not r["ocr_required"]],
        "multi_turn": [r for r in scored if r["type"] == "multi_turn"],
    }
    page_meta_available = any(p is not None for _s, p in index_info["chunk_keys"])

    # 相关性判定分三种口径，全部为 question 级计算后宏平均（§1.1.7）。
    #   page_level   : 命中且页码等于 expected_page（最严格；旧代码无页码 → 不适用）
    #   source_level : 命中且来源等于 expected_source（四组可比；单文档语料下区分度低）
    #   answer_kw    : 命中片段文本包含 expected_keywords（与页码/来源无关，衡量"真的召回到答案"）
    metrics_page_level = (
        {name: _metrics_for(rows_, relevant_counts_page, key="page_relevant")
         for name, rows_ in subsets.items()}
        if page_meta_available else None)
    metrics_source_level = {
        name: _metrics_for(rows_, relevant_counts_source, key="source_relevant", with_recall=False)
        for name, rows_ in subsets.items()}
    metrics_answer_kw = {
        name: _metrics_for(rows_, relevant_counts_kw, key="kw_relevant", with_recall=False)
        for name, rows_ in subsets.items()}

    page_correct = []
    for row in scored:
        if row["expected_page"] is None:
            continue
        top = next((d for d in row["ranked"] if d["source_relevant"]), None)
        if top is not None and top.get("page") is not None:
            page_correct.append(1.0 if top["page"] == row["expected_page"] else 0.0)
    page_accuracy = {
        "n": len(page_correct),
        "accuracy": round(sum(page_correct) / len(page_correct), 4) if page_correct else None,
    }

    repo_root = ROOT
    payload = {
        "group": args.group,
        "codebase_root": str(repo_root),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "env": {k: os.environ.get(k) for k in
                ["RERANK_METHOD", "RERANK_MODEL_NAME", "OCR_ENABLED", "HYBRID_ALPHA", "HF_HUB_OFFLINE"]},
        "config": {
            "RERANK_METHOD": getattr(config, "RERANK_METHOD", None),
            "RERANK_MODEL_NAME": getattr(config, "RERANK_MODEL_NAME", None),
            "OCR_ENABLED": getattr(config, "OCR_ENABLED", None),
            "HYBRID_ALPHA": getattr(config, "HYBRID_ALPHA", None),
            "CHUNK_SIZE": getattr(config, "CHUNK_SIZE", None),
            "RETRIEVAL_TOP_K": getattr(config, "RETRIEVAL_TOP_K", None),
            "RERANK_TOP_K": getattr(config, "RERANK_TOP_K", None),
        },
        "index": index_info,
        "page_metadata_available": page_meta_available,
        "metrics": {
            "definition": {
                "page_level": "命中片段来源==expected_source 且 page==expected_page；Recall 分母=该页 chunk 数",
                "source_level": "命中片段来源==expected_source（不校验页码）",
                "answer_keyword": "命中片段来源==expected_source 且文本包含任一 expected_keywords（不管页码）",
                "aggregation": "question 级计算后宏平均（WBS §1.1.7）；应拒答题不计入任何指标分母",
            },
            "page_level": metrics_page_level,
            "source_level": metrics_source_level,
            "answer_keyword": metrics_answer_kw,
            "page_accuracy_on_hit": page_accuracy,
        },
        "abstain_items": [r["question_id"] for r in rows if r["should_abstain"]],
        "questions": rows,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{args.group}] chunks={index_info['total_chunks']} docs={len(index_info['documents'])}")
    print(f"[{args.group}] all(source) : {metrics_source_level['all']}")
    print(f"[{args.group}] scan(source): {metrics_source_level['scanned']}")
    print(f"[{args.group}] all(kw)     : {metrics_answer_kw['all']}")
    print(f"[{args.group}] scan(kw)    : {metrics_answer_kw['scanned']}")
    if metrics_page_level:
        print(f"[{args.group}] all(+page)  : {metrics_page_level['all']}")
        print(f"[{args.group}] scan(+page) : {metrics_page_level['scanned']}")
    else:
        print(f"[{args.group}] page-level  : N/A (no page metadata in this codebase)")
    print(f"[{args.group}] wrote: {out_path}")


if __name__ == "__main__":
    main()
