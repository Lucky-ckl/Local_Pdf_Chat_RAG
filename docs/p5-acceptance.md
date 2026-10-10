# P5 评测体系 · 验收记录

> 依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》§8 P5（706–807 行）、§1.1.7（宏平均口径）与 §13 G5。
> 记录原则：只记录**实际执行并取得证据**的结论；`BLOCKED` / `NOT VERIFIED` 明确标注，不写成通过。

## 1. 范围与依赖

- 任务：P5.1–P5.7。依赖 P1–P4 全部可运行（已具备，HEAD `425cf9f`）。
- 环境：Windows；Python 3.11.7（`.venv`）；CPU（无 GPU）；`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- **LLM 后端限制**：本机无 SiliconFlow/Magick API Key，Ollama 不可达；按批准约束**不提供新的 LLM 后端**。
  因此所有需要真实生成的验收项（P5.3、TTFT 等）标记 `BLOCKED`，**不使用替身结果冒充真实 LLM 指标**。
- 批准约束兑现：不安装 `ragas`；不下载模型；不调用付费 API；不上传任何公司文档；未提交/推送 Git。

## 2. 产出物

| 产出物 | 路径 | 说明 |
| --- | --- | --- |
| 评测集 | `eval/eval_set.json` | 26 条，6 类，字段齐备（`human_verified` 全 `false`，待人工复核） |
| 评测集生成脚本 | `eval/make_eval_set.py` | 声明式生成，可重复产出同一评测集 |
| 检索指标脚本 | `eval/run_eval.py` | 只跑检索；三种相关性口径；结果带时间戳 |
| 延迟分解脚本 | `eval/collect_latency.py` | 阶段 P50/P95；单页解析+OCR 单列 |
| 四组驱动 | `eval/run_groups.ps1` | ①–④ 同集同参 |
| 延迟驱动 | `eval/run_latency.ps1` | ④ 与 ① |
| 复核辅助表 | `eval/REVIEW.md` | 逐题机器预检，供人工复核 |
| 评测报告 | `eval/REPORT.md` | P5.6，六项齐备 |
| 原始结果 | `eval/results/20261010-143204_{baseline,rerank_only,ocr_only,full}/retrieval.json` | 四组检索 |
| 原始结果 | `eval/results/20261010-141235_{full,baseline}/latency.json` | 延迟 |
| 合成扫描样本 | `eval/samples/scan_maintenance_notice.pdf`（+ `make_scan_sample.py`） | 纯图像 PDF，验证 OCR 链路 |
| 语料 | `eval/corpus/` | 挖掘机案例 PDF + 合成扫描件 |

## 3. 逐任务验收

### P5.1 评测集构建 — PASS

- 规模 26 条（WBS 要求 20–30）；6 类全覆盖（simple_fact 6 / model_code 4 / fuzzy_semantic 4 / multi_turn 4 / scanned 5 / abstain 3，应拒答 ≥3 ✓）。
- 多轮题成对：q015、q017 为上下文轮，q016、q018 为含 `history` 的追问轮。
- 字段齐备；每条含 `human_verified` 状态（全部 `false`）。
- 未用应拒答 3 条校准任何阈值（本阶段不设阈值）。
- 应拒答前提已核验：语料 chunk 文本中 `油耗/机油/品牌/额定工作压力/兆帕/压力` 命中数均为 0。
- 证据：`eval/eval_set.json`、`eval/REVIEW.md`、上述文本核查命令。

### P5.2 检索指标脚本 — PASS

- 产出 `eval/run_eval.py`；含 `Hit@k`/`MRR@k`/`Recall@k`（k=3/5/10），question 级计算后**宏平均**（§1.1.7），报告含 `n_questions` 与 k。
- **只跑检索不跑生成**：脚本不导入/不调用任何 LLM 客户端。
- 结果写入 `eval/results/<时间戳>_<组>/`，多次运行产生不同时间戳目录，不覆盖历史。
- **重复运行稳定性已实测**：两次独立全量运行（tag `20261010-140506` 与 `20261010-143204`）四组的 top-10 排名序列**逐题完全一致**。
- 相关性口径说明：`page_level`（source+page）/`source_level`（source）/`answer_keyword`（source+关键词）；旧代码无页码时 `page_level` 记 `null`（不记 0）。详见 `eval/REPORT.md` §2。

### P5.3 生成忠实度判定 — BLOCKED

- 依赖真实 LLM 生成回答后才能逐题判定「回答是否被引用内容支持」；本机无可用 LLM 后端，无法生成，故**未能开展人工判定**。
- `ragas` 为可选附加项，本轮未安装、未接入（无 LLM 亦无法计算）。
- 阻塞原因、所需条件、对 G5 的影响：见 `eval/REPORT.md` §5。
- **未以替身输出的分数冒充真实忠实度指标。**

### P5.4 改造前后对比 — PASS

- 四组同集同参（`HYBRID_ALPHA=0.7`/`CHUNK_SIZE=400`/`RETRIEVAL_TOP_K=10`/`RERANK_TOP_K=5`），仅切 OCR 与重排：
  ① 改造前基线；② 仅修重排；③ 仅加 OCR；④ 全量改造。
- **① 由改造前代码真实产出**：`git archive 38b39f6` 导出到仓库外 `%TEMP%\rag_p5_baseline`，独立建索引（7 chunk，无 OCR）。
  **偏差记录**：WBS 原文建议 `git stash` / 独立 worktree；因当前工作区含大量未提交改动（P0–P4）不可 stash，经用户批准改用 `git archive` 到仓库外，未改变当前 Git 状态、未切分支。
- 扫描件子集**单独报告**（n=5）：0.0（①②）→ 1.0（③④）——OCR 增益清晰且未被总分掩盖。
- 关键数字：非扫描件 `answer_keyword MRR@5` 0.7407（①）→ 0.8907（④）；全体 `page_level MRR@5` 0.7188（②）→ 0.9362（④）；`page_accuracy_on_hit` 0.913（④，21/23）。
- 全部未命中项可归因于「无 LLM → 改写回退」（q016/q018）。
- 证据：`eval/results/20261010-143204_*/retrieval.json`、`eval/REPORT.md` §3。

### P5.5 延迟与成本统计 — PARTIAL PASS（离线可测部分）/ 关键项 BLOCKED

- 已取得（P50/P95，n=130）：查询改写（回退）0.003/0.177 ms；嵌入+FAISS 10.898/12.954 ms；BM25 0.333/0.410 ms；RRF 0.028/0.033 ms；**Rerank 5761.109/6058.192 ms**；检索总计 5770.422/6070.085 ms（④）与 26.436/29.261 ms（①，降级路径）。
- 文档解析 + OCR **单页单列**：扫描件 2681 ms（冷，含引擎初始化）/ 2419 ms（热 P50）；文本层 PDF 26.3 / 31.7 ms 每页。
- **TTFT / 真实查询改写 / 全量生成：BLOCKED**（无 LLM 后端）。表中「检索总计」只是首字之前的检索耗时，**不是 TTFT**，不得当作 TTFT 使用。
- 证据：`eval/results/20261010-141235_{full,baseline}/latency.json`、`eval/REPORT.md` §4。

### P5.6 评测报告 — PASS

- `eval/REPORT.md` 含六项：评测集说明、指标定义、四组对比、延迟表、未达标项、已知局限。
- 局限部分非空且具体：评测集规模小（26 条/2 文档/8 chunk）、OCR 仅合成单页（倾斜/手写未验证）、重排模型与 WBS 默认不一致、rerank CPU 延迟高、`HYBRID_ALPHA` 未校准、多轮改写未验证、`human_verified` 待复核、P0 遗留项。

### P5.7 P5 验收 — G5 判定

| G5 条件 | 判定 | 证据 |
| --- | --- | --- |
| 评测报告产出 | **PASS** | `eval/REPORT.md`（六项齐备） |
| 改造前后同集同参对比成立，且 ① 由旧代码产出 | **PASS** | `eval/results/20261010-143204_*`；①由 `38b39f6` 导出代码产出（7 chunk、扫描件 0 chunk） |
| TTFT 有前后对比数字 | **FAIL（BLOCKED）** | 无可用 LLM 后端，无法测量真实 TTFT（见 `eval/REPORT.md` §5） |

**G5 最终判定：NOT SATISFIED（部分满足）** —— 3 项 Gate 条件中 2 项满足，`TTFT 前后对比` 因外部依赖缺失而阻塞。
按批准约束，不通过降低验收标准（例如用替身 TTFT）来使 G5 通过。补齐条件：提供可用 LLM 后端（或明确授权云端 API），即可完成 P5.3 与 P5.5 的 TTFT/改写/全量生成，从而满足 G5。

## 4. 未完成 / 待人工确认

- `human_verified`：全部仍为 `false`，需人工按 `eval/REVIEW.md` 逐条复核后置 `true`。
- 默认重排模型 `BAAI/bge-reranker-base` 未加载验证；实测使用本地缓存 `BAAI/bge-reranker-v2-m3`（离线加载成功，偏差已记录）。
- `RERANK_METHOD=llm` 未真实验证；`HYBRID_ALPHA` 未校准。
- OCR 对倾斜/手写/低分辨率真实扫描件未验证。
- 应拒答行为端到端（生成阶段）未验证。
- P0 遗留：端到端真实问答耗时 BLOCKED；远程 CI 未验证；`OCR_MAX_PAGES=50` 待确认。

## 5. Git 与文件保护

- 本轮**未执行** `git add/commit/push/reset/clean/stash`，**未切换分支**；P0–P4 全部未提交改动保持原样。
- 受保护 WBS `Local_Pdf_Chat_RAG_WBS技术执行文档_v1.0.md` 未被修改（SHA-256 复核为
  `9D43A27848C681B3F70B7BB9EB55B0ED1AE834B0EB75034CB182946473C482F5`）。
- 本轮新增均为**未跟踪**文件：`eval/**`。`eval/index/`、`eval/results/` 属本地产物，
  未被 `.gitignore` 覆盖（仓库现仅忽略 `data/index/`）——**提交前需人工决定是否忽略或排除**。
