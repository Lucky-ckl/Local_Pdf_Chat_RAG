# P5 评测报告（Local_Pdf_Chat_RAG）

> 依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》§8（P5，约 706–807 行）与 §1.1.7（宏平均口径）。
> 记录原则：**只写实际跑出来的数字**；未执行/受限项一律标注 `BLOCKED` / `NOT VERIFIED`，不用替身结果冒充真实模型指标。

- 生成时间：2026-10-10（本机时区 Asia/Shanghai）
- 代码库：`D:\Project\RAG-Project\Local_Pdf_Chat_RAG`，分支 `main`，HEAD `425cf9f`（P2.1）
- 改造前基线代码：`git archive 38b39f6` 导出到仓库外 `%TEMP%\rag_p5_baseline`（该提交无 `core/ocr.py`、`core/reranker.py` 用 bi-encoder 冒充 CrossEncoder）
- 环境：Windows，Python 3.11.7（`.venv`），CPU（无 GPU），`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`
- 主要结果目录：
  - 四组检索：`eval/results/20261010-143204_{baseline,rerank_only,ocr_only,full}/retrieval.json`
  - 延迟：`eval/results/20261010-141235_{full,baseline}/latency.json`
  - 人工复核辅助表：`eval/REVIEW.md`

> **结果目录说明**：`eval/results/` 下另有 `20261010-134933`、`-135826`、`-140506` 三个较早的时间戳目录，
> 属**指标口径定稿前**的中间运行（当时 `answer_keyword` 未限定来源、`page_level` 对无页码的基线误记 0）。
> 按「不得覆盖历史结果」的要求全部保留；**本报告的权威数字一律取自 `20261010-143204`（四组检索）与
> `20261010-141235`（延迟）**。`eval/results/.last_ts` 是早期草稿脚本留下的陈旧文件，未被现行脚本使用。

---

## 1. 评测集说明（P5.1）

- 产出物：`eval/eval_set.json`，由 `eval/make_eval_set.py` 生成，**26 条**。
- 语料（`eval/corpus/`）：`挖掘机维修案例(样例）.pdf`（4 页，可提取文本）＋ `scan_maintenance_notice.pdf`（**合成 image-only PDF**，1 页，无文本层，需 OCR）。入库后共 **8 个 chunk**（挖掘机 7 + 扫描件 1）。
- 类型覆盖（WBS 要求的 6 类，全部覆盖）：

| 类型 | 条数 | 题号 |
| --- | --- | --- |
| 简单事实 simple_fact | 6 | q001–q006 |
| 型号/编号 model_code | 4 | q007–q010 |
| 模糊语义 fuzzy_semantic | 4 | q011–q014 |
| 多轮指代 multi_turn | 4 | q015–q018（q015/q017 为上下文轮，q016/q018 为含 `history` 的追问轮） |
| 扫描件 scanned | 5 | q019–q023（`ocr_required=true`） |
| 应拒答 abstain | 3 | q024–q026（`should_abstain=true`） |

- 字段齐备（WBS §8 P5.1）：`question_id / question / history / expected_source / expected_page / expected_keywords / should_abstain / ocr_required / human_verified`。
- **`human_verified` 仍全部为 `false`**：按 WBS 与指令要求，只有人工实际复核后才能置 `true`，本报告不代为判定。已生成 `eval/REVIEW.md` 提供逐题机器预检（期望来源/页码/关键词是否进入 top-10），供人工复核后手工置位。
- **未使用应拒答的 3 条校准任何阈值**：本阶段不设阈值，不做阈值校准。
- 应拒答项前提已核验：`eval/index/full/chunks.json` 的全部 chunk 文本中，`油耗 / 升每小时 / 机油 / 品牌 / 额定工作压力 / 兆帕 / MPa / 压力` 出现次数均为 **0**，即 q024–q026 所问信息确实不在语料内。
- **真实扫描件与合成样本的区分**：`scan_maintenance_notice.pdf` 是**合成的纯图像 PDF**，用于验证扫描件 OCR 流程，**不是**真实业务扫描件；OCR 对倾斜、手写、低分辨率真实件的效果未验证（见 §6）。

## 2. 指标定义（P5.2）

脚本 `eval/run_eval.py`：**只做检索，不跑生成**（不调用任何 LLM）。k 取 3/5/10；所有指标按 **question 级计算后宏平均**（§1.1.7）；报告同时给出有效样本数 `n_questions` 与 k。应拒答题不计入任何指标分母，仅单列于 `abstain_items`。

命中判定用三种口径（同一份结果同时给出，避免单一口径误导）：

| 口径 | 定义 | Recall 分母 |
| --- | --- | --- |
| `page_level` | 命中片段 `source==expected_source` **且** `page==expected_page` | 该期望页的 chunk 数 |
| `source_level` | 命中片段 `source==expected_source`（不校验页码） | —（不单列 Recall） |
| `answer_keyword` | 命中片段 `source==expected_source` 且文本包含任一 `expected_keywords`（不校验页码） | —（不单列 Recall） |

- `Hit@k`：前 k 是否至少有一个相关片段；`MRR@k`：第一个相关片段排名的倒数；`Recall@k`：前 k 命中相关片段数 / 该题相关片段总数。
- 另有 `page_accuracy_on_hit`：在已命中期望来源的题目中，**首个来源相关片段的页码**等于期望页的比例。
- 子集：`all`（23）/`non_scanned`（18）/`scanned`（5）/`multi_turn`（4）。
- 结果按时间戳写入 `eval/results/<时间戳>_<组>/retrieval.json`，**不覆盖历史结果**。

### 为什么同时给三种口径

语料只有 2 个文档、每题期望来源唯一，`source_level` 几乎必然饱和（见 §3 的 1.0），区分度低；旧代码（①）没有页码元数据，`page_level` 对其**不适用**（记 `null`，不记 0，避免把「旧代码无法表达页码」误判成「检索完全失败」）。因此跨组对比同时看 `answer_keyword`（与页码无关、可跨组比较）与 `page_level`（最严格，仅在能输出页码的组之间比较）。

## 3. 四组同集同参对比（P5.4）

四组使用**同一评测集、同一语料、同一参数**（`HYBRID_ALPHA=0.7`（默认，未由 env 覆盖）、`CHUNK_SIZE=400`、`RETRIEVAL_TOP_K=10`、`RERANK_TOP_K=5`），仅切换 OCR 与重排开关；①独立建索引（旧代码）。

| 组 | 代码库 | OCR | Rerank | 入库 chunk |
| --- | --- | --- | --- | --- |
| ① baseline | `38b39f6` 导出（改造前） | 关 | 旧实现（离线加载失败 → 降级） | 7 |
| ② rerank_only | 当前 | 关 | `cross_encoder` / `BAAI/bge-reranker-v2-m3` | 7 |
| ③ ocr_only | 当前 | 开 | `none`（真实 RRF） | 8 |
| ④ full | 当前 | 开 | `cross_encoder` / `BAAI/bge-reranker-v2-m3` | 8 |

### 3.1 全体（all，n=23）

| 指标 | ① baseline | ② rerank_only | ③ ocr_only | ④ full |
| --- | --- | --- | --- | --- |
| answer_keyword Hit@3 | 0.6522 | 0.7391 | 0.9565 | 0.9565 |
| answer_keyword Hit@5 | 0.7391 | 0.7826 | 0.9565 | **1.0** |
| answer_keyword Hit@10 | 0.7826 | 0.7826 | 0.9565 | **1.0** |
| answer_keyword MRR@5 | 0.5797 | 0.6971 | 0.8551 | **0.9145** |
| page_level Hit@3 | N/A | 0.7391 | 0.9565 | 0.9565 |
| page_level Hit@5 | N/A | 0.7826 | 0.9565 | **1.0** |
| page_level MRR@5 | N/A | 0.7188 | 0.9058 | **0.9362** |
| page_level Recall@5 | N/A | 0.7174 | 0.9348 | 0.9348 |
| source_level Hit@3/5/10 | 0.7826 | 0.7826 | 1.0 | 1.0 |
| page_accuracy_on_hit | N/A | 0.8889 (n=18) | 0.8696 (n=23) | 0.913 (n=23) |

### 3.2 扫描件子集（scanned，n=5，单列）

| 指标 | ① baseline | ② rerank_only | ③ ocr_only | ④ full |
| --- | --- | --- | --- | --- |
| source_level Hit@k | 0.0 | 0.0 | **1.0** | **1.0** |
| answer_keyword Hit@k | 0.0 | 0.0 | **1.0** | **1.0** |
| page_level Hit@k | N/A | 0.0 | **1.0** | **1.0** |

**OCR（P2）增益**：扫描件子集 0.0 → 1.0（k=3/5/10 全指标）。无 OCR 的两组（①、②）扫描件 0 chunk 入库，该子集必然全挂，与 WBS 预期一致。

### 3.3 非扫描件子集（non_scanned，n=18）

| 指标 | ① baseline | ② rerank_only | ③ ocr_only | ④ full |
| --- | --- | --- | --- | --- |
| answer_keyword Hit@3 | 0.8333 | 0.9444 | 0.9444 | 0.9444 |
| answer_keyword Hit@5 | 0.9444 | 1.0 | 0.9444 | **1.0** |
| answer_keyword MRR@5 | 0.7407 | 0.8907 | 0.8148 | **0.8907** |
| page_level Hit@3 | N/A | 0.9444 | 0.9444 | 0.9444 |
| page_level Hit@5 | N/A | 1.0 | 0.9444 | **1.0** |
| page_level MRR@5 | N/A | 0.9185 | 0.8796 | **0.9185** |

**Rerank（P1）增益**：在非扫描件子集上，开启重排把 `answer_keyword MRR@5` 从 0.7407（①）提升到 0.8907（④），`Hit@5` 0.9444 → 1.0；`page_level MRR@5` 从 0.8796（③，无重排）提升到 0.9185（④）。② 与 ④ 数值相同，说明该子集的提升由**重排**带来（OCR 对其无影响）。

### 3.4 多轮子集（multi_turn，n=4）

| 指标 | ① baseline | ② rerank_only | ③ ocr_only | ④ full |
| --- | --- | --- | --- | --- |
| answer_keyword Hit@3 | 0.75 | 0.75 | 1.0 | 0.75 |
| page_level Hit@3 | N/A | 0.75 | 1.0 | 0.75 |
| page_level Hit@5 | N/A | 1.0 | 1.0 | 1.0 |

**重要限制**：多轮子集**不能**用来验证 P3.2 的改写能力。项目当前无可用 LLM 后端，`history` 存在时改写会尝试并失败，随后**回退原 query**（日志：`未配置 SiliconFlow API Key` → `查询改写失败，回退原 query`；探针显示 `rewrite_used=False`）。因此 q016/q018 实际是用代词原句「那最后更换的是什么零件？」直接检索。④ 在 k=3 未命中而 ③ 命中，属重排对「仅代词 query」的排序差异，不是能力回退的结论。

### 3.5 未命中项的归因

- ④ `answer_keyword Hit@3 = 0.9565`，唯一未命中项为 **q018**；`page_accuracy_on_hit = 0.913`（21/23）的两个偏差项为 **q016、q018**。
- 该两题**正好是两处代词追问**（依赖改写）。q016 的期望页在 top-3 内（第 3 位），但**首个来源相关片段**落在 p4（正确页排在其后），故被计入页码偏差；q018 的期望页与关键词都落在 k=3 之外（top-10 内仍可命中）。即：**所有未命中与页码偏差都可由「无 LLM → 改写回退」解释**，其余 21 题的三种口径全部命中。

## 4. 延迟与成本（P5.5）

`eval/collect_latency.py`，`n=130`（26 题 × 5 轮），先做一次预热查询以排除模型冷启动。单位 ms。

| 阶段 | 组 | P50 | P95 |
| --- | --- | --- | --- |
| 查询改写 | ④ full | 0.003 | 0.177 |
| 嵌入 + FAISS 检索（semantic） | ④ full | 10.898 | 12.954 |
| BM25 检索 | ④ full | 0.333 | 0.410 |
| RRF 融合（hybrid_merge） | ④ full | 0.028 | 0.033 |
| **Rerank** | ④ full | **5761.109** | **6058.192** |
| 检索总计 | ④ full | **5770.422** | **6070.085** |
| 检索总计 | ① baseline | 26.436 | 29.261 |

**文档解析 + OCR（单页单列）**：

| 文件 | 页数 | OCR 页 | 冷启动 总/单页 | 热启动 P50 单页 |
| --- | --- | --- | --- | --- |
| `scan_maintenance_notice.pdf`（纯图像） | 1 | 1 | 2681 ms / 2681 ms | 2419 ms |
| `挖掘机维修案例(样例）.pdf`（有文本层） | 4 | 0 | 105 ms / 26.3 ms | 31.7 ms |

入库一次性开销（8 chunk）：解析+OCR ≈ 5660 ms，嵌入 ≈ 7545 ms（含模型加载）。

**解读（真实测量口径说明）**：
- ④ 的检索延迟几乎完全由 **Rerank** 决定（P50 5.76 s / 查询，CPU，`bge-reranker-v2-m3` 对 10 个候选逐对打分）；语义检索仅约 11 ms、BM25 <1 ms、RRF <0.1 ms。
- ① baseline 的 26 ms 是**降级路径**：旧代码的「CrossEncoder」模型离线加载失败（日志：`sentence-transformers/distiluse-base-multilingual-cased-v2 does not appear to have a file named pytorch_model.bin`）→ 直接回退，因此**不能**把 26 ms 当作「旧代码的正常重排延迟」。
- OCR 约 **2.4 s/页**（纯图像页），是入库期的主要成本；含文本层的页仅约 30 ms/页。
- **TTFT（生成首 token）与全量生成无法测量**：见 §5。上表「检索总计」只是**首字之前**的检索耗时，**不等于 TTFT**（TTFT 还需加真实生成时间），不得当作 TTFT 数字使用。

## 5. 未达标项与阻塞（BLOCKED）

| 项 | 状态 | 原因 | 所需条件 | 对 G5 的影响 |
| --- | --- | --- | --- | --- |
| P5.3 生成忠实度（人工逐题判定「回答是否被引用内容支持」） | **BLOCKED** | 项目当前无可用 LLM 后端（无 SiliconFlow/Magick Key，Ollama 不可达），无法生成真实回答，也就无从判定其忠实度 | 提供可用 LLM 后端（或明确授权调用云端 API、接受费用） | P5.3 无法完成 → G5 的对应项不满足 |
| P5.3 可选 `ragas` 指标 | **BLOCKED**（未实施） | 未安装 `ragas`（本轮禁止安装），且其指标仍依赖 LLM（含评判模型） | 授权安装并配置 LLM | 同上；属可选附加项 |
| P5.5 TTFT（流式首 token） | **BLOCKED** | 无可用 LLM 后端，无法产生真实 token 流 | 同上 | **G5「TTFT 有前后对比数字」不满足** |
| P5.5 查询改写耗时（真实触发） | **BLOCKED** | 无 LLM，改写必然失败并回退（`rewrite_used=False`），表内 0.003 ms 只是失败回退耗时 | 同上 | 属 P5.5 分解项之一，未取得真实数字 |
| P5.5 全量生成耗时 | **BLOCKED** | 同上 | 同上 | 同上 |
| 应拒答行为端到端验证 | **NOT VERIFIED** | 拒答发生在生成阶段，检索脚本不跑生成 | 可用 LLM 后端 | 不影响 G5 的检索对比结论 |

## 6. 已知局限（诚实披露）

1. **评测集规模小、语料小**：26 条问题、2 个文档、8 个 chunk；单用户本地 CPU 场景，结论不具备规模外推性。
2. **OCR 覆盖不全**：只用**合成的纯图像 PDF** 单页验证了扫描链路；对**倾斜、手写、印章、低分辨率、多栏排版**的真实扫描件未验证。合成样本的 OCR 文本本身存在误识（如 `HPU-01` 被识为 `H PU－O1`、`400-816-2233` 含全角字符），关键词型题目对此有一定容忍度，但不能据此推断真实件可用。
3. **重排模型与 WBS 默认不一致（偏差记录）**：WBS §8 指定默认 `BAAI/bge-reranker-base`；本机**未验证**该模型的加载，实测使用本地已缓存并离线加载成功的 **`BAAI/bge-reranker-v2-m3`**（`RERANK_MODEL_NAME` 经配置传入，非硬编码）。因此本报告的「重排增益」结论绑定于 `bge-reranker-v2-m3`，**不能**等同于 `bge-reranker-base` 的表现。`RERANK_METHOD=llm` 亦未真实验证。
4. **重排延迟偏高**：CPU 无 GPU 下 rerank 约 5.76 s/查询，显著拉高响应时间；未做量化、批处理或候选裁剪优化（属后续阶段）。
5. **`HYBRID_ALPHA=0.7` 未校准**：沿用默认值，P5 未做 α 调优；本报告不声称该取值最优。
6. **多轮改写未验证**：无 LLM 导致改写回退，故 multi_turn 子集**不构成** P3.2 改写能力的证据（见 §3.4）。
7. **`source_level` 口径饱和**：单文档语料下该口径普遍为 1.0，区分度低，仅供参考。
8. **`human_verified` 全为 false**：评测集尚未经人工复核，期望值可能仍有个别偏差。
9. **P0 遗留（未解决）**：端到端真实问答耗时 BLOCKED；远程 CI 未验证；`OCR_MAX_PAGES=50` 默认值待确认。
10. **可复现性边界**：检索指标为确定性结果；延迟为单机单次采样（P50/P95 基于 130 个样本），受机器负载影响，绝对值仅供同机对比。

## 7. 复现命令

```powershell
# 0) 环境
$env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'

# 1) 改造前基线代码（不改动当前 Git 状态 / 不切分支）
git archive 38b39f6 | tar -x -C $env:TEMP\rag_p5_baseline

# 2) 评测集
.\.venv\Scripts\python.exe eval\make_eval_set.py

# 3) 四组同集同参检索对比（自动写入 eval\results\<时间戳>_<组>\retrieval.json）
pwsh -NoProfile -File eval\run_groups.ps1

# 4) 延迟分解（full 与 baseline）
pwsh -NoProfile -File eval\run_latency.ps1 -Repeats 5

# 5) 人工复核辅助表
.\.venv\Scripts\python.exe eval\make_review.py --result eval\results\<时间戳>_full\retrieval.json
```

## 8. 结论

- **已成立**：改造前后**同集同参**对比成立，且 ① 基线**由改造前代码（`38b39f6` 导出）真实产出**；OCR 对扫描件的增益（0.0 → 1.0）与重排对排序精度的增益（非扫描件 MRR@5 0.7407 → 0.8907）均有可复现命令与原始 JSON 支撑。
- **未成立**：P5.3 生成忠实度、P5.5 的 TTFT/改写/全量生成因**无可用 LLM 后端**而 BLOCKED；因此 **G5 判定为部分满足（TTFT 对比数字缺失）**，详见 `docs/p5-acceptance.md`。
