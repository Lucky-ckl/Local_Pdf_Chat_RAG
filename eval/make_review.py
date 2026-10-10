"""生成人工复核辅助表（WBS v1.0 §8 P5.1：每条评测项须有 human_verified 状态）。

本脚本**不做人工判定**，只把机器可核查的事实摊开，降低人工复核成本：
- 期望来源 / 期望页码 / 期望关键词；
- 机器预检：期望来源是否进入 top-10、期望页是否出现在 top-10、哪些期望关键词真的
  出现在命中的 chunk 文本里；
- top-1 的来源 / 页码 / 片段预览。

用法：
    python eval\make_review.py --result eval\results\<ts>_full\retrieval.json --out eval\REVIEW.md
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_eval  # noqa: E402


def _cell(text):
    return (text or "").replace("|", "\\|").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--eval-set", default="eval/eval_set.json")
    ap.add_argument("--out", default="eval/REVIEW.md")
    args = ap.parse_args()

    import json
    payload = json.loads(pathlib.Path(args.result).read_text(encoding="utf-8"))
    eval_set = {it["question_id"]: it for it in run_eval._load_eval_set(args.eval_set)}
    group = payload.get("group")
    ts = payload.get("generated_at")

    lines = [
        "# 评测集人工复核辅助表（机器预检，非人工结论）",
        "",
        f"- 依据：WBS v1.0 §8 P5.1（每条评测项须有 `human_verified` 状态）。",
        f"- 来源结果：`{args.result}`（group=`{group}`，generated_at=`{ts}`）。",
        "- **本表不代表人工复核已完成**：`human_verified` 仍为 `false`，需由人工逐条确认后手动置 `true`。",
        "- 机器预检列说明：`来源✓` = 期望来源进入 top-10；`页✓` = 期望页码出现在 top-10；"
        "`关键词命中` = 在 top-10 任一命中片段文本中找到的期望关键词（空表示未找到）。",
        "",
        "| 题号 | 类型 | 问题 | 期望来源 | 期望页 | 期望关键词 | 来源✓ | 页✓ | 关键词命中 | top-1 来源/页 | 人工复核结论 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for qid, item in eval_set.items():
        row = next((r for r in payload["questions"] if r["question_id"] == qid), None)
        if row is None:
            continue
        top10 = row["ranked"]
        exp_src = run_eval._basename(item.get("expected_source"))
        exp_page = item.get("expected_page")
        kws = item.get("expected_keywords") or []
        src_ok = "✓" if any(d["source"] == exp_src for d in top10) else "✗"
        if item.get("should_abstain"):
            src_ok = page_ok = "—（应拒答）"
            found = "—"
        else:
            page_ok = "✓" if any(d["page"] == exp_page for d in top10) else "✗"
            found = "、".join(sorted({w for d in top10 for w in d["keyword_hit"]})) or "（无）"
        top1 = top10[0] if top10 else {}
        lines.append(
            "| {qid} | {typ} | {q} | {src} | {page} | {kw} | {s} | {p} | {f} | {t1} |  |".format(
                qid=qid, typ=item.get("type"), q=_cell(item.get("question")),
                src=_cell(item.get("expected_source")), page=exp_page if exp_page is not None else "—",
                kw=_cell("、".join(kws)) or "—", s=src_ok, p=page_ok, f=_cell(found),
                t1=f"{top1.get('source','')} / p{top1.get('page')}"))
    lines.append("")
    out = pathlib.Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(eval_set)} items)")


if __name__ == "__main__":
    main()
