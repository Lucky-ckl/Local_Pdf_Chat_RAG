"""P5.5 延迟分解（WBS v1.0 §8 P5.5）。

离线可测量（本机 CPU，无 GPU，无云端 LLM）：
- 文档解析 + OCR：单页耗时单列（冷启动含 OCR 引擎初始化；热启动为引擎就绪后的单页耗时）
- 嵌入 + FAISS 检索（retriever trace 的 ``semantic`` 阶段）
- BM25 检索（``bm25`` 阶段）
- RRF 融合（``hybrid_merge`` 阶段）
- Rerank（``rerank`` 阶段）
- 单次查询检索总耗时

无法离线测量（项目当前无可用 LLM 后端）→ 明确标记 BLOCKED：
- 查询改写（仅在多轮有历史时触发；无 LLM Key/Ollama 时回退原 query）
- 生成首 token（TTFT）
- 全量生成

结果按时间戳目录写出，不覆盖历史。用法：

    $env:OCR_ENABLED="true"; $env:RERANK_METHOD="cross_encoder"
    .\.venv\Scripts\python.exe eval\collect_latency.py --group full --repeats 5 `
        --index-dir eval\index\full --out eval\results\<ts>_full\latency.json
"""

import argparse
import datetime
import json
import pathlib
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_eval  # noqa: E402  （同目录脚本，提供入证与检索封装）


def _pctl(values, q):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return round(vals[0], 3)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return round(vals[lo] * (1 - frac) + vals[hi] * frac, 3)


def _summarize(values):
    return {
        "n": len(values),
        "P50": _pctl(values, 0.50),
        "P95": _pctl(values, 0.95),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "min": round(min(values), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


def _measure_parse(corpus_dir):
    """单页解析 + OCR 耗时：冷启动（含引擎初始化）与热启动各测一次/多次。"""
    import core.document_loader as loader
    import config

    out = {"ocr_enabled": bool(getattr(config, "OCR_ENABLED", True)), "files": []}
    load_document = getattr(loader, "load_document", None)
    if load_document is None:
        out["available"] = False
        out["note"] = "改造前代码无按页 load_document（无页码/无 OCR），单页解析耗时对该代码库不适用"
        return out
    out["available"] = True
    for path in sorted(pathlib.Path(corpus_dir).iterdir()):
        if path.suffix.lower() != ".pdf":
            continue
        cold = None
        try:
            t0 = time.perf_counter()
            blocks = load_document(str(path))
            cold = (time.perf_counter() - t0) * 1000.0
            pages = len(blocks)
            ocr_pages = sum(1 for b in blocks if getattr(b, "ocr_used", False))
        except Exception as exc:  # 记录而非掩盖
            out["files"].append({"file": path.name, "error": str(exc)})
            continue
        warm = []
        for _ in range(3):
            t0 = time.perf_counter()
            load_document(str(path))
            warm.append((time.perf_counter() - t0) * 1000.0)
        out["files"].append({
            "file": path.name,
            "pages": pages,
            "ocr_pages": ocr_pages,
            "cold_total_ms": round(cold, 3),
            "cold_ms_per_page": round(cold / max(pages, 1), 3),
            "warm_ms_per_page": _pctl([w / max(pages, 1) for w in warm], 0.50),
            "warm_total_ms": _summarize(warm),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--corpus", default="eval/corpus")
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import config

    eval_set = run_eval._load_eval_set(args.eval_set)
    corpus = run_eval._discover_corpus(args.corpus)
    index_info = run_eval._build_index(corpus, args.index_dir)
    parse_stats = _measure_parse(args.corpus)

    # 预热：第一次查询包含查询侧模型/缓存冷启动，不计入统计
    run_eval._retrieve("预热查询", None)

    stage_samples = {}
    total_samples = []
    rewrite_used = 0
    for rep in range(args.repeats):
        for it in eval_set:
            t0 = time.perf_counter()
            _c, _d, _m, trace = run_eval._retrieve(it["question"], it.get("history"))
            total_samples.append((time.perf_counter() - t0) * 1000.0)
            stages = (trace.get("retrieval_trace") or {}).get("stages_ms") or {}
            for name, ms in stages.items():
                stage_samples.setdefault(name, []).append(float(ms))
            if trace.get("rewrite_used"):
                rewrite_used += 1

    stages = {name: _summarize(vals) for name, vals in sorted(stage_samples.items())}

    payload = {
        "group": args.group,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "repeats": args.repeats,
        "n_questions": len(eval_set),
        "codebase_root": str(ROOT),
        "env": {k: __import__("os").environ.get(k) for k in
                ["RERANK_METHOD", "RERANK_MODEL_NAME", "OCR_ENABLED", "HF_HUB_OFFLINE"]},
        "config": {
            "OCR_ENABLED": getattr(config, "OCR_ENABLED", None),
            "RERANK_METHOD": getattr(config, "RERANK_METHOD", None),
            "RERANK_MODEL_NAME": getattr(config, "RERANK_MODEL_NAME", None),
        },
        "index": {"total_chunks": index_info["total_chunks"], "modern": index_info["modern"],
                  "timing_ms": index_info["timing_ms"]},
        "parse_and_ocr": parse_stats,
        "stages_available": bool(stages),
        "stages_ms": stages,
        "retrieval_total_ms": _summarize(total_samples),
        "rewrite_triggered_samples": rewrite_used,
        "blocked_stages": {
            "query_rewrite_llm": "BLOCKED：项目当前无可用 LLM 后端（无 SiliconFlow Key / Ollama 不可达），多轮改写回退原 query，无法测量真实改写耗时",
            "ttft": "BLOCKED：无可用 LLM 后端，无法测量真实生成首 token 延迟",
            "full_generation": "BLOCKED：无可用 LLM 后端，无法测量全量生成耗时",
        },
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.group}] parse+ocr files={len(parse_stats['files'])}")
    print(f"[{args.group}] stages={list(stages)}")
    print(f"[{args.group}] total P50={payload['retrieval_total_ms']['P50']} P95={payload['retrieval_total_ms']['P95']}")
    print(f"[{args.group}] wrote: {out_path}")


if __name__ == "__main__":
    main()
