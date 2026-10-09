# 现状缺陷基线（baseline-defects.md）

> 本文件由 **P0.1** 建立骨架，**P0.4** 完成逐条落档（文件:行号 + 复现方式 + 观察结果）。
> 记录原则：只记录**已复核**的事实；未复核内容标注为「待验证」。不得把推测写成已确认。
> 复核环境：Windows / Python 3.11.7；仓库基线提交 `38b39f6`；复核时间 2026-10-09。

## 状态取值

- `已确认`：已在本仓库复核并定位到文件与行号，有可复现证据。
- `未复现`：按描述尝试但未能复现（需注明原因）。
- `待验证`：尚未复核。
- `已修复`：已修复并有对应验证（后续阶段填写）。

## 缺陷清单

### D1 — reranker 用 bi-encoder 冒充 CrossEncoder

- **描述**：`core/reranker.py` 用 sentence-transformers 的 `CrossEncoder` 类加载 **bi-encoder** 模型
  `sentence-transformers/distiluse-base-multilingual-cased-v2`；该模型并非交叉编码器。加载/推理失败时静默回退为
  「按原序返回」并给出**位置伪分数**；README 宣称 supports a CrossEncoder。
- **文件:行号**：
  - `core/reranker.py:29-31`（`CrossEncoder('.../distiluse-base-multilingual-cased-v2')`）
  - `core/reranker.py:34-36`（加载异常 → `_cross_encoder = None`）
  - `core/reranker.py:46-48`（不可用 → `logging.warning` + `_fallback_results`）
  - `core/reranker.py:118-121`（`_fallback_results`：按原序返回，分数 `1.0 - idx/len(docs)`）
  - `README.md:44`、`README.zh-CN.md:44`（宣称支持 CrossEncoder）
- **复现命令 A（模型类型）**：
  `python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('sentence-transformers/distiluse-base-multilingual-cased-v2'); print(c.architectures, c.model_type, c.num_labels)"`
- **实际观察 A**：`['DistilBertModel'] distilbert 2` —— 架构为 DistilBERT 双塔基座，非交叉编码器。
- **复现命令 B（降级为原序）**：
  `python -c "import core.reranker as r; r.get_cross_encoder=lambda: None; print(r.rerank_with_cross_encoder('q',['docA','docB','docC'],['c1','c2','c3'],[{},{},{}],top_k=5))"`
- **实际观察 B**：返回 `[('c1', score 1.0), ('c2', score 0.667), ('c3', score 0.333)]` —— 顺序未变，分数为按位置伪分数。
- **状态**：**已修复**（P1.1–P1.2，见下方「修复验证」）。以上「复现命令/实际观察」描述的是**修复前**行为。

**修复验证（P1.2）**
- 模型来源改为 `config.RERANK_MODEL_NAME`（默认 `BAAI/bge-reranker-base`，真实交叉编码器），不再硬编码
  `sentence-transformers/distiluse-base-multilingual-cased-v2`；`core/reranker.py` 以 `CrossEncoder(RERANK_MODEL_NAME)` 加载。
- 失败不再静默：加载/预测失败输出含**模型名与异常类型**的日志并置位 `rerank_unavailable`（P1.1）。
- 真实重排验证（本地缓存模型 `BAAI/bge-reranker-v2-m3`，离线）：同一 query + 同一候选，重排前后 top-5 顺序确实变化（2 样本，见 `docs/decisions.md` §7.2）。
- **待验证**：配置默认值 `BAAI/bge-reranker-base` 尚未在本地真实加载（HF 缓存缺失）；上述真实验证使用的是
  `BAAI/bge-reranker-v2-m3`，二者不可混为一谈。README 的 CrossEncoder 措辞已于 P0.4 更新。

### D2 — 多轮对话未真正生效

- **描述**：`core/generator.py` 的 `query_answer` 无 `history` 参数；UI 累积的历史从未传入模型，多轮对话实际未生效（UI 文案不实）。
- **文件:行号**：
  - `core/generator.py:221`（`def query_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None)`，无 `history`）
  - `rag_demo.py:337-363`（`process_chat` 维护 `history` 仅用于 UI）
  - `rag_demo.py:356`（`query_answer(question, enable_web_search, model_choice_val)`，未传历史）
  - `api_router.py:116`（REST 调用同样不传历史）
- **复现命令**：`python -c "import inspect; from core.generator import query_answer; print(list(inspect.signature(query_answer).parameters))"`
- **实际观察**：`['question', 'enable_web_search', 'model_choice', 'progress']` —— 无 `history`。
- **状态**：已确认

### D3 — 流式为死代码

- **描述**：`core/generator.py` 的 `stream_answer()` 全项目无调用方；云端请求体写死 `"stream": False`。
  该函数即便被调用，云端分支也是「整体生成后一次 yield」，并非真正流式。
- **文件:行号**：
  - `core/generator.py:270`（`def stream_answer(...)`）
  - `core/generator.py:71`（云端 payload `"stream": False`）
  - 另见 `core/generator.py:154`、`core/generator.py:254`、`core/reranker.py:79`（其他 `"stream": False`）
- **复现命令**：
  `findstr /s /n /c:"stream_answer" *.py core\*.py features\*.py utils\*.py tests\*.py`
  `findstr /s /n /c:"\"stream\": False" core\*.py`
- **实际观察**：`stream_answer` 仅 `core/generator.py:270` 一处（定义，无调用）；`"stream": False` 命中 `core/generator.py:71/154/254` 与 `core/reranker.py:79`。
- **状态**：已确认

### D4 — 索引仅存内存

- **描述**：`core/vector_store.py` / `core/bm25_index.py` 索引仅存内存，无落盘/加载，进程重启即丢失。
- **文件:行号**：
  - `core/vector_store.py:93-161`（`VectorStore` 仅持有内存结构，无 save/load/`write_index`）
  - `core/bm25_index.py:25-63`（`BM25IndexManager` 仅内存，无持久化）
  - `core/vector_store.py:156-161` / `core/bm25_index.py:59-63`（`clear()` 仅清内存）
- **复现命令**：`findstr /s /n /c:"write_index" /c:"read_index" /c:"INDEX_DIR" core\*.py`
- **实际观察**：无匹配 —— 无任何持久化实现。`config.INDEX_DIR` 于 P0.3 新增，但当前尚无代码使用它。
- **状态**：已确认

### D5 — 扫描件 PDF 未支持

- **描述**：`core/document_loader.py` 仅用 pdfminer 读取 PDF 文本层，无 OCR；无文本层的扫描件返回空串，上层 `rag_demo.py` 抛 `ValueError`。
- **文件:行号**：
  - `core/document_loader.py:28-33`（PDF 分支仅 `extract_text_to_fp`，无 OCR）
  - `rag_demo.py:67-69`（`text = extract_text(...)`；`if not text: raise ValueError("文档内容为空或无法提取文本")`）
- **复现命令**：`python -c "from core.document_loader import extract_text; print(len(extract_text('挖掘机维修案例(样例）.pdf')))"`
- **实际观察**：示例 PDF（含文本层）返回 2025 字符。对**无文本层**的扫描件 PDF，本函数返回空串并触发上层 `ValueError`；本仓库未附扫描件样本，复现需自备扫描件 PDF。
- **状态**：已确认（扫描件 → 空串 → 上层抛错为代码既定路径；样例 PDF 为文本型）

### D6 — 无评测 / trace / 索引持久化

- **描述**：仓库无评测代码、无 trace、无索引持久化。
- **文件:行号**：不适用（「缺失」类缺陷）。
- **复现命令**：`git ls-files | findstr /i "eval trace manifest"`；并确认无 `eval/` 目录。
- **实际观察**：无 `eval/` 目录；无 trace / manifest 相关源码或配置。分别对应 P5（评测）与 P2.1（持久化）。
- **状态**：已确认

### D7 — openwiki 文档与代码漂移

- **描述**：`openwiki/` 自动生成文档与真实代码存在漂移：记录**不存在的测试名**；内部链接以 `/` 开头（相对仓库根），在 `openwiki/` 目录内解析会指向不存在路径（失效内链）。
- **文件:行号**：
  - `openwiki/interfaces/rest-api.md:446`（记录测试名 `test_status_endpoint()`）
  - `openwiki/index.md`、`openwiki/quickstart.md` 等（链接形如 `/configuration/environment-and-models.md`、`/core/document-processing.md`）
- **复现命令**：`findstr /s /n /c:"test_status_endpoint" openwiki\*.md openwiki\*\*.md`
- **实际观察**：`tests/` 下实际 API 测试名为 `test_status_reports_current_version_without_credentials`；不存在 `test_status_endpoint`。内链以 `/…` 开头而文件实际位于 `openwiki/…`，属失效内链。对应 P6.1（openwiki 处理）。
- **状态**：已确认

### D8 — 死代码

- **描述**：`core/generator.py` 导入 `evaluate_source_credibility` 但从未使用。
- **文件:行号**：
  - `core/generator.py:21`（`from features.conflict_detector import detect_conflicts, evaluate_source_credibility`）
  - `features/conflict_detector.py:35`（定义 `evaluate_source_credibility`）
- **复现命令**：`python -c "import pathlib; s=pathlib.Path('core/generator.py').read_text(encoding='utf-8'); print('usage excluding import:', s.count('evaluate_source_credibility')-1)"`
- **实际观察**：`usage excluding import: 0` —— 导入后从未被使用。
- **状态**：已确认

---

## P0.5 基线运行数字

> 由 **P0.5** 填写；均注明环境（`.venv` 还是系统 Python）与真实命令。**未把系统 Python 的结果冒充 `.venv` 的结果。**
> 环境：`.venv`（Python 3.11.7，`python -m pip install -r requirements-dev.txt` 安装成功）。

- **测试（`.venv`）**：
  - `pytest -q` → `26 passed`
  - `pytest --cov --cov-report=term-missing`（`.coveragerc`：source=core,config, fail_under=60）→ `26 passed`，覆盖率 **62.62%**（达标，退出码 0）
- **启动（`.venv`）**：加载 `rag_demo.py` 的 Gradio `Blocks` 并以 `server_port=17995` 启动，`GET http://127.0.0.1:17995/ → HTTP 200`，随后正常关闭。
- **首次灌库耗时（`.venv`，冷启动；文档 `挖掘机维修案例(样例）.pdf`，2025 字符 → 7 chunks，embedding 384 维）**：
  - `extract_text=0.170s` / `split_text≈0.000s` / `encode_texts=21.209s`（含 embedding 模型冷加载）/ `FAISS build≈0.000s` / `BM25 build=0.622s`
  - **首次灌库合计 ≈ 22.001s**
- **单次问答耗时**：**BLOCKED（部分）**
  - 完整问答需 LLM 后端；本机 **Ollama 不可达**（`http://localhost:11434/api/tags` 超时），**未执行任何云端 API 调用**，故未测完整问答端到端耗时。
  - 可测的**检索段**（同一 `.venv` 进程内，问题「挖掘机液压泵故障如何排查」）：`query_encode=0.0093s` / `FAISS=0.0001s` / `BM25=0.0002s` / `hybrid_merge≈0.0000s` → **检索合计 ≈ 0.0096s**。
  - 后续：待具备可用 LLM 后端后补测完整问答耗时（并区分首 token 与总耗时）。

---

## 备注

- 本轮（P0.4）仅**记录**基线缺陷，不修复 D1–D8 中属于 P1/P2 的缺陷。
- 新增改造参数 `OCR_MAX_PAGES` 的默认值未由 WBS 指定，暂取 50，见 `docs/decisions.md`。
