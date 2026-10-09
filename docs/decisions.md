# 实现决策记录（decisions.md）

> 本文件由 **P0.1** 建立骨架。后续每个阶段在此追加实现决策。
> 记录原则：只记录**已经发生**的事实；尚未发生的提交与未实现的功能一律标注为「计划」，不得写成已完成。

## 1. 项目来源与基线

- 本仓库 **Fork 自** 上游 `weiwill88/Local_Pdf_Chat_RAG`（教学型参考实现）；上游原有代码与文档归属上游作者。
- 当前分支：`main`。
- 基线提交：`38b39f6`（`docs: add generated OpenWiki code documentation (#56)`）——这是**唯一**提交，即上游快照。
- **本仓库尚无任何改造提交**（此为事实，非计划）。
- 本地路径：`D:\Project\RAG-Project\Local_Pdf_Chat_RAG`。
- 远端 `origin` 目前仍指向上游 `weiwill88/Local_Pdf_Chat_RAG`；本仓库暂**无自有远端地址**。

## 2. 改造范围与总体计划

改造依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》，按 `P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7` 串行推进；前置阶段未通过 Gate 不得进入下游阶段。

| 阶段 | 目标 | 状态 |
| --- | --- | --- |
| P0 | 基线与契约冻结 | 进行中（P0.1–P0.6 已实施，G0 待确认） |
| P1 | 检索链修复 | 已完成（P1.1–P1.6 已实施，G1 待确认） |
| P2 | 索引持久化与解析层（含 OCR） | 计划 |
| P3 | 会话与生成链 | 计划 |
| P4 | 前端重写 | 计划 |
| P5 | 评测体系 | 计划 |
| P6 | 交付与部署 | 计划 |
| P7 | 可选增强 | 计划 |

> 上表状态仅反映**实际**进度；未完成项一律标为「计划」。

## 3. 提交规范

- 每项改造**单独成一次 commit**，message 说明「改了什么 + 为什么」。
- 不把多类不相关的改动混入同一提交。
- 提交前复核工作区，避免把密钥或无关文件纳入。
- 不删除既有测试、不把改造项从清单中移除。

提交计划（**均为计划，尚未提交**；实际提交以 `git log` 为准）：

| 提交主题（计划） | 涉及文件（计划） | 对应 WBS 任务 |
| --- | --- | --- |
| `fix:` 安全与异常处理清理 | `.gitignore`、`rag_demo.py` | P0.2（工作区保护） |
| `test:` 测试环境隔离 | `tests/test_api_status.py` | P0.2（测试基线） |
| `docs:` 增加 Fork 来源声明与改造清单占位 | `README.md`、`README.zh-CN.md`、`docs/decisions.md`、`docs/baseline-defects.md` | P0.1 |
| `chore:` 配置集中化（既有超参 + 新增参数环境变量化） | 待定（P0.3 范围） | P0.3 |
| `docs:` 现状缺陷基线落档（D1–D8，含文件行号与复现） | `docs/baseline-defects.md` | P0.4 |
| `chore:` 测试与 CI 基线（覆盖率阈值） | 待定（P0.6 范围） | P0.6 |
| `fix:` 检索链修复（重排 / 混合分数 / 降级契约） | 待定（P1 范围） | P1 |

> 上表「涉及文件」为提交时的**预期**范围；实际提交内容与顺序以 `git log` 为准，本表不表示任何提交已完成。

## 4. 与 WBS v1.0 描述不一致、需要更新的旧描述

1. **项目路径不一致**：WBS 写作 `D:/AI_Project/Local_Pdf_Chat_RAG`，实际为 `D:\Project\RAG-Project\Local_Pdf_Chat_RAG`。
2. **P0.2 的未提交改动描述已过期**：WBS 称「当前工作区存在一处未提交改动（`config.py` 的 `MAGICK_MODEL_NAME` 默认值）」，但实际 `config.py` 工作区是干净的；当前实际未提交改动共 7 项，完整清单见第 5 节。需更新 WBS 或本文件口径。
3. **P0.1 徽章处置方式**：WBS 要求徽章「改为本仓库地址或移除」。因本仓库暂无自有远端，本次采取**移除上游 CI/Release/Stars 徽章**并**保留必要的来源归属**（不使用伪造的本仓库地址）。
4. **运行基线（已更新）**：`.venv` 已由 `requirements-dev.txt` 重建（Python 3.11.7），`pytest` 与启动均以 `.venv` 验证；问答端到端耗时因本机无可用 LLM 后端（Ollama 不可达）暂 **BLOCKED**（见 `docs/baseline-defects.md`）。
5. **新增参数 `OCR_MAX_PAGES` 默认值待确认**：WBS v1.0 未指定其默认值，本仓库暂取 `50` 并在 `config.py` 标注「待确认」，待明确后更新。

## 5. 本地未提交改动（截至 P0.6 完成，均未提交）

以下改动存在于工作区，尚未提交（以 `git status` 为准的完整清单）：

**安全 / 调试 / 测试隔离 / 来源标注（P0.1–P0.2 早期）**
- `.gitignore`：新增忽略规则 `/test.py`；新增索引持久化目录 `data/index/`。
- `rag_demo.py`：修正单文件处理失败时的异常处理（不再因单个文件失败中断整批，且不向外回显异常详情）。
- `tests/test_api_status.py`：用 `monkeypatch` 隔离本机 `.env` 中的真实凭证，使「未配置凭证」用例可独立复现。
- `README.md` / `README.zh-CN.md`：P0.1——Fork 来源声明与「尚未实现」改造清单占位；移除上游 CI/Release/Stars 徽章；上游统计表改写为「上游项目（来源参考）」；P0.4——把重排序描述改为与已确认缺陷 D1 一致的措辞。

**P0.3 配置集中化**
- `config.py`：既有超参改为环境变量可覆盖（`_env_int` / `_env_float` / `_env_bool`）；新增 8 个改造参数。
- `example.env`：与 `config.py` 逐项对应的示例项（默认以注释给出）。
- `tests/test_config.py`：新增环境变量转换/回退、新参数存在性、覆盖行为测试。

**P0.4 缺陷基线**
- `docs/baseline-defects.md`：D1–D8 逐条补齐「文件:行号 + 复现命令 + 实际观察」（全部「已确认」）。

**P0.5 可运行环境与基线**
- `.venv`：由 `requirements-dev.txt` 重建（不进 Git）。
- `docs/baseline-defects.md`：记录首次灌库与检索段实测数字。

**P0.6 测试与 CI 基线**
- `.coveragerc`（新增）：覆盖率统计范围（core+config）与阈值 `fail_under = 60`。
- `pytest.ini`：补充覆盖率基线说明（不改 `addopts`，保证无 pytest-cov 环境仍可运行）。
- `tests/test_text_splitter.py`、`tests/test_vector_store.py`、`tests/test_document_loader.py`：补齐基线测试，使 core+config 覆盖率 ≥ 60%。

**P1 检索链修复（P1.1–P1.5）**
- `core/reranker.py`：P1.1 失败可见化（异常日志含模型名/异常类型、置位 `rerank_unavailable`）；P1.2 模型名由 `config.RERANK_MODEL_NAME` 提供（不再硬编码）；P1.3 请求级状态 `RerankStatus(score_source, rerank_disabled_by_user, rerank_unavailable)` 随 `RerankResult` 返回。
- `core/retriever.py`：P1.4 `hybrid_merge` 改用真实 FAISS 距离归一化（`sim(d)=1/(1+max(d,0))`）；重排未生效/失败时改用真实 RRF 融合；读取请求级状态而非模块级全局。
- `core/vector_store.py`：新增 `search_with_scores`（返回真实平方 L2 距离）；`search` 仍返回原三元组。
- `config.py` / `example.env`：新增 `RERANK_METHOD`、`RERANK_MODEL_NAME`（P1.2）。
- `rag_demo.py`：P1.5 `get_system_models_info` 改为从 config / 运行时来源读取。
- 新增测试：`tests/test_reranker.py`、`tests/test_rerank_contract.py`、`tests/test_hybrid_merge.py`、`tests/test_system_models_info.py`。

- 历史遗留：本地临时脚本 `test.py` 曾含硬编码 API Key，现已从磁盘移除；其 Key 已在签发方吊销（未在此记录任何密钥内容）。

## 6. P0 各任务实测状态（截至本次）

| 任务 | 交付 | 实测证据 | 状态 |
| --- | --- | --- | --- |
| P0.1 来源标注与仓库基线 | `README*` + `docs/` | 上游徽章已移除；改造清单 7 项未勾选 | 已完成 |
| P0.2 分支与提交规范 | `docs/decisions.md` §3 | 提交计划已列（均为计划，未提交） | 已完成（计划） |
| P0.3 配置集中化 | `config.py` + `example.env` + 测试 | `pytest` 全绿；环境变量覆盖/转换/回退有测试 | 已完成 |
| P0.4 现状缺陷基线 | `docs/baseline-defects.md` | D1–D8 全部「已确认」，D1–D3 附可复现命令 | 已完成 |
| P0.5 依赖与运行基线 | `.venv` + 基线数字 | `pytest` 26 passed；Gradio `HTTP 200`；首次灌库 ≈ 22.0s | 部分完成（问答耗时 BLOCKED） |
| P0.6 测试与 CI 基线 | `.coveragerc` + `pytest.ini` + 测试 | `--cov` 报告 **62.62%**（≥60%），退出码 0 | 已完成（远程 CI 未运行） |

> 状态以本地实测为准；**远程 CI 未实际运行**，不得视为已通过。

## 7. P1 验收记录（P1.6）

> 由 **P1.6** 编写。记录 P1.1–P1.5 的实现、验收证据、测试结果与已知限制；状态以**本地实测**为准。
> **模型命名（重要）**：本节的真实验证使用本地 HF 缓存中可用的交叉编码器 `BAAI/bge-reranker-v2-m3`（离线，
> `HF_HUB_OFFLINE=1`）；而 `config.RERANK_MODEL_NAME` 的默认值是 `BAAI/bge-reranker-base`，该默认值
> **尚未在本地真实加载验证**（缓存缺失）。二者不得混为一谈。
> 复核环境：`.venv`（Python 3.11.7）；时间 2026-10-09。

### 7.1 各任务实现与证据

| 任务 | 交付 | 关键实现 | 证据 | 状态 |
| --- | --- | --- | --- | --- |
| P1.1 Reranker 失败可见化 | `core/reranker.py` 错误处理 | 加载/预测失败 → `logging.error`（含模型名、异常类型）+ 置位 `rerank_unavailable`；不中断问答 | `tests/test_reranker.py`（caplog 断言模型名/异常类型/告警） | PASS |
| P1.2 真 CrossEncoder 接入 | 模型替换 | 模型名走 `config.RERANK_MODEL_NAME`（默认 `BAAI/bge-reranker-base`），不再硬编码；保留 `RERANK_METHOD=llm` 备选 | §7.2 真实重排（模型 `BAAI/bge-reranker-v2-m3`）；`tests/test_reranker.py` | PASS（默认模型真实加载见 §7.7） |
| P1.3 重排降级契约 | `score_source` 语义 | `score_source ∈ {"rerank","rrf"}`；`RerankStatus(score_source, rerank_disabled_by_user, rerank_unavailable)` 随 `RerankResult` 请求级传递 | `tests/test_rerank_contract.py`；§7.3 | PASS |
| P1.4 混合分数可信化 | `core/retriever.py::hybrid_merge` | 语义分数用真实 FAISS 平方 L2 距离归一化 `sim(d)=1/(1+max(d,0))`（注释含公式/理由/局限）；BM25 按当次最大值归一到 [0,1] | `tests/test_hybrid_merge.py`；§7.4 | PASS |
| P1.5 模型信息去硬编码 | `rag_demo.py::get_system_models_info` | 分块/α/重排信息读取 config；嵌入模型读 `core.embeddings.EMBED_MODEL_NAME`；重排描述随 `RERANK_METHOD` 动态变化 | `tests/test_system_models_info.py` | PASS |

### 7.2 重排前后 top-5 顺序对比（2 样本）

- 方法：7 条固定中文语料；每个 query 先经 `hybrid_merge`（α=0.7）取 top-5 作为「重排前」，再经
  `recursive_retrieval(..., max_iterations=1)`（跨真实调用路径）得到「重排后」。重排使用本地缓存的真实交叉编码器
  `BAAI/bge-reranker-v2-m3`；`RETRIEVAL_TOP_K=10`、`RERANK_TOP_K=5`。
- 语料：d1 液压泵异常噪音/吸油滤芯排查；d2 发动机机油压力过低；d3 液压系统压力不稳/溢流阀；
  d4 空调不制冷；d5 液压泵尖锐啸叫/吸入空气；d6 挖掘机行走马达无力；d7 定期更换液压油与滤芯。

**样本 1 — query「液压泵异响怎么排查」**

| 排名 | 重排前（hybrid_merge） | 重排后（cross-encoder） |
| --- | --- | --- |
| 1 | d6 | d1 |
| 2 | d1 | d5 |
| 3 | d7 | d7 |
| 4 | d5 | d3 |
| 5 | d2 | d6 |

变化：d1 升至第 1；d3 进入 top-5；d2 掉出 top-5。

**样本 2 — query「空调不制冷是什么原因」**

| 排名 | 重排前（hybrid_merge） | 重排后（cross-encoder） |
| --- | --- | --- |
| 1 | d4 | d4 |
| 2 | d5 | d2 |
| 3 | d3 | d3 |
| 4 | d6 | d7 |
| 5 | d7 | d1 |

变化：d2 升至第 2；d1 进入 top-5；d5、d6 掉出 top-5。

### 7.3 重排不可用 / 用户关闭时的降级表现

- **服务故障**（`get_cross_encoder()` 返回 `None`）：`recursive_retrieval` 最终顺序 = **RRF 融合顺序**，而非候选原序。
  样本 1：RRF = `d7,d1,d6,d5,d2`，而重排前候选顺序为 `d6,d1,d7,d5,d2`（二者不同 → 证明执行的是 RRF 融合）；
  日志两条 WARNING（「重排服务不可用…」「…已回退到 RRF 融合排序」）；状态 `(rrf, False, True)`；问答不中断。
- **用户关闭**（`RERANK_METHOD=none`）：同样使用 RRF 融合顺序，但日志为 INFO（「重排已由用户关闭…」）、**无告警**；
  状态 `(rrf, True, False)`。与「服务故障」可明确区分。
- **重排调用抛异常**：`retriever` 捕获后同样降级为 RRF 并告警（覆盖于 `tests/test_rerank_contract.py`）。

### 7.4 混合检索分数归一化前后对比

- 旧：语义分数 = `1.0 - i/n`（仅依赖候选排名序位，与真实相似度无关）。
- 新：语义分数 = `sim(d) = 1/(1+max(d,0))`，`d` 为 FAISS 平方 L2 距离（越小越相似），输出 (0,1] 且随距离**单调递减**。

**样本 1 的语义候选**（doc / 平方 L2 距离 / 旧伪分数 / 新分数）：

| doc | 距离 | 旧 1-i/n | 新 1/(1+d) |
| --- | --- | --- | --- |
| d1 | 0.8834 | 1.0000 | 0.5309 |
| d7 | 0.9780 | 0.8571 | 0.5056 |
| d2 | 1.0524 | 0.7143 | 0.4872 |
| d3 | 1.1261 | 0.5714 | 0.4703 |
| d6 | 1.1481 | 0.4286 | 0.4655 |
| d5 | 1.1504 | 0.2857 | 0.4650 |
| d4 | 1.1545 | 0.1429 | 0.4641 |

**样本 2 的语义候选**：

| doc | 距离 | 旧 1-i/n | 新 1/(1+d) |
| --- | --- | --- | --- |
| d4 | 0.1276 | 1.0000 | 0.8868 |
| d5 | 0.4984 | 0.8571 | 0.6674 |
| d3 | 0.6979 | 0.7143 | 0.5890 |
| d6 | 0.8445 | 0.5714 | 0.5421 |
| d7 | 0.8982 | 0.4286 | 0.5268 |
| d1 | 0.9428 | 0.2857 | 0.5147 |
| d2 | 1.0055 | 0.1429 | 0.4986 |

关键差异（样本 1）：d5 与 d6 的距离几乎相等（1.1504 vs 1.1481），旧伪分数却相差很大（0.2857 vs 0.4286），
而新分数几乎相同（0.4650 vs 0.4655）——说明旧分数与真实相似度脱节、新分数与距离单调对应。

### 7.5 测试结果（真实命令与结果）

- `pytest -q`（`.venv`, Python 3.11.7）→ **69 passed**（最终通过数），0 failed，约 14s。
- 口径说明：**61** 为 **P1.5 之前**的通过数（已包含 P1.1–P1.4 新增用例）；P1 阶段整体新增 **41** 个用例
  （`test_reranker.py` 9 + `test_rerank_contract.py` 12 + `test_hybrid_merge.py` 12 + `test_system_models_info.py` 8）；
  61 + 8（P1.5 新增）= 69（最终通过数）。
- `pytest -q --cov` → **70.48%**（阈值 60%），退出码 0。
- P1 相关测试（`test_reranker.py` + `test_rerank_contract.py` + `test_hybrid_merge.py` + `test_system_models_info.py`）→ **41 passed**。
- `python -m py_compile`（本轮修改文件）→ 退出码 0；`git diff --check` → 干净。

> 上述命令输出取自本会话的真实执行记录；**本次文档收尾（P1 验收后）未重新执行测试**，故不声称本轮重新验证。

### 7.6 P1 验收条件与 G1

| # | P1.6 验收条件 | 结果 |
| --- | --- | --- |
| 1 | 重排前后 top-5 顺序对比（≥2 样本） | PASS（§7.2） |
| 2 | 重排不可用时的降级表现记录 | PASS（§7.3） |
| 3 | 混合检索分数归一化前后对比说明 | PASS（§7.4） |
| 4 | `pytest` 全绿 | PASS（69 passed） |
| 5 | 消费 `baseline-defects.md` 中的 D1 已标记「已修复」 | PASS（D1 已修复，见该文件） |

**G1（重排前后排序确实变化；重排不可用时仍能回答且状态可见）**：PASS —— 2 样本顺序均变化；
降级为真实 RRF 融合且 WARNING 可见、问答不中断。

### 7.7 已知限制与未完成（不得视为已通过）

- **模型命名**：真实验证用本地缓存模型 `BAAI/bge-reranker-v2-m3`；配置默认值 `BAAI/bge-reranker-base`
  **尚未在本地真实加载验证**（HF 缓存缺失）。两者不可混为一谈。
- `RERANK_METHOD=llm` 分支未做真实验证（本机 Ollama 不可达）。
- `HYBRID_ALPHA=0.7` 未经评测校准（计划 P5 校准）。
- 归一化 `sim(d)=1/(1+d)` 在小距离区间分辨率偏低（样本 1 分数集中在 [0.464, 0.531]）；对候选集合变化稳定，但压缩了近距离区分度。
- `core/reranker.py::_fallback_results` 仍附位置伪分数；但产品级结果由 `retriever` 覆盖为 RRF 分数并标记
  `score_source="rrf"`，不作为语义/重排分数对外。
- P0 遗留（保留）：单次问答端到端耗时 **BLOCKED**（Ollama 不可达、未调用云端 API）；远程 CI 未验证；
  `OCR_MAX_PAGES=50` 默认值待确认。
- 上述取证脚本在临时目录运行，未提交仓库。

