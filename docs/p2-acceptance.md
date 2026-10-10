# P2 索引持久化与解析层（含 OCR）· 验收记录

> 依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》§5 P2（约 376–504 行）。
> 记录原则：只记录**已验证**事实；未验证/受限项明确标注，不写成已通过。

## 1. 范围与依赖

- 任务：P2.1–P2.9；依赖 P0.3（配置集中化）、P2.1（索引持久化）。
- 环境：Python 3.11.7（`.venv`）；Windows；CPU（无 GPU）。

## 2. 改动 / 新增文件

- `core/document_loader.py`：P2.2 按页 Block；P2.5 扫描件判定（§1.1.3）与 OCR 接线、页面级低文本补足。
- `core/ocr.py`（新增）：P2.3 RapidOCR 引擎（懒加载+缓存、首次/后续耗时、页数截断）；P2.4 PDF 页渲染（PyMuPDF）；P2.6 阅读顺序整理。
- `core/text_splitter.py`：P2.7 `Chunk` 结构与 `split_blocks`（按页切分、跨页取起始页）。
- `rag_demo.py`：P2.7 入库接线改为 `load_document` + `split_blocks`，chunk 元数据补 `page/chunk_index/ocr_used`。
- `requirements.txt`：新增 `rapidocr_onnxruntime`、`PyMuPDF`（WBS P2.3 / P2.4）。
- 测试：`tests/test_ocr.py`（新增）、`tests/test_document_loader.py`、`tests/test_text_splitter.py`、`tests/test_index_persistence.py`。

## 3. 逐任务验收

### P2.2 Loader 改造为按页 Block

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| `load_document(filepath) -> list[Block]` | PASS | `core/document_loader.py::load_document`（:41）返回 `list[Block]`；`Block{text,page,source,ocr_used}`（:25）。测试 `test_load_document_pdf_pages_start_at_one`、`test_load_document_pdf_pages_are_independent`、`test_load_document_non_pdf_returns_single_page_zero_block`。 |
| PDF 页码从 1 开始 | PASS | `_load_pdf_blocks` 用 `pdfminer.high_level.extract_pages` 逐页、`enumerate(..., start=1)`；`test_load_document_pdf_pages_start_at_one` 断言页码 `[1, 2, 3]`。 |
| 非 PDF `page=0` | PASS | `test_load_document_non_pdf_returns_single_page_zero_block`（`.txt` → 单 Block，`page=0`，`ocr_used=False`）。 |
| 空白页保留真实页数 | PASS | `test_load_document_pdf_blank_page_keeps_real_page_count`（中间空白页仍占一页，页码 `[1, 2, 3]`）。 |
| `extract_text` 兼容 | PASS | `extract_text`（:174）改为 `"\n".join(block.text ...)` 兼容包装；原 5 个 `extract_text` 用例未改动且继续通过，另新增 `test_extract_text_still_wraps_pdf_pages`。 |
| 不支持的扩展名返回空文本 | PASS | `test_load_document_unsupported_extension_returns_empty_block`（`.bin` → 空文本 Block）。 |

### P2.3 OCR 引擎接入

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| `core/ocr.py` 提供 `ocr_image(image) -> OcrResult` | PASS | `core/ocr.py::ocr_image`；单测 `test_ocr_image_orders_lines_and_reports_engine`。 |
| 二次调用不重新初始化引擎 | PASS | `test_engine_is_lazy_loaded_and_cached`：patch `RapidOCR` 计数，两次调用仅实例化 1 次。 |
| 超限截断并告警 | PASS | `test_truncate_pages_warns_and_limits`、`test_ocr_pages_truncates_before_engine_call`（`OCR_MAX_PAGES` 告警）。 |

- 引擎：`rapidocr_onnxruntime` 1.4.4；模型随 wheel 内置（`.../rapidocr_onnxruntime/models/*.onnx`），**完全离线可跑**，无独立模型下载。

### P2.4 PDF 页渲染

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 能对指定页渲染出位图 | PASS | `test_render_pdf_pages_renders_real_page`：PyMuPDF 1.28.2，DPI=150，返回 HxWx3 `ndarray`。 |
| 加密/损坏页告警且不中断整篇 | PASS | `test_render_pdf_pages_marks_failed_page_and_continues`：第 2 页 `render_failed`，其余页继续。 |

### P2.5 扫描件自动判定

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 判定按「每页平均字符数」（§1.1.3） | PASS | `is_scanned_document`；`test_scan_detection_uses_per_page_average`（总量同、页数不同 → 判定不同）。 |
| 文本型 / 扫描型 / 混合型 / 空文档四类 | PASS | `test_scan_detection_text_document_is_not_scanned`、`_scanned_document`、`_mixed_200_pages_one_scan`、`_empty_document_is_not_scanned`。 |
| 混合型（1 页扫描）只对扫描页 OCR | PASS | E2E（合成 image-only PDF 样本）：文本页（2661 字符）+ 图像页 → 整篇非扫描件；`ocr_used=[false, true]`（仅第 2 页 OCR）。 |
| 页面级低文本页补足并记日志 | PASS | `test_load_document_low_text_page_triggers_page_level_ocr`（3 页仅第 2 页 OCR，`ocr_used=[false,true,false]`）。 |
| `force_ocr` 跳过判定 | PASS | `test_load_document_force_ocr_overrides_detection`。 |

### P2.6 阅读顺序恢复

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 按 y 排序、同一行内按 x 排序 | PASS | `order_lines`；`test_order_lines_sorts_by_y_then_x`。 |
| 相邻行按 y 间距合并为段落 | PASS | `lines_to_text`；`test_lines_to_text_merges_vertically_stacked_lines_only`（并排行不被拼接，避免跨栏错拼）。 |
| 排序逻辑注释说明阈值依据 | PASS | `core/ocr.py::PARAGRAPH_Y_RATIO` 注释说明 0.6 倍行高取值依据。 |

### P2.7 chunk 页码元数据

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| chunk 含 `source/page/chunk_index/ocr_used` | PASS | `core/text_splitter.py::Chunk`、`split_blocks`；`test_split_blocks_assigns_page_and_chunk_index`。 |
| 跨页 chunk 记录起始页 | PASS | `test_split_blocks_cross_page_chunk_takes_start_page`。 |
| 页码抽样 ≥5 个与原文一致 | PASS | E2E（临时目录实测）6/6 正确（3 页 × 2 chunk）。 |
| 元数据可持久化（与 P2.1 兼容） | PASS | `test_chunk_page_metadata_roundtrips_through_persistence`。 |

### P2.8 OCR 单测

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 新增 ≥8 用例 | PASS | `tests/test_ocr.py` 新增 13 例 + `test_index_persistence.py` 元数据往返 1 例。 |
| `pytest` 全绿 | PASS | 见 §5。 |
| 覆盖率达标（≥60%） | PASS | 见 §5。 |

### P2.9 P2 验收

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 重启进程后问答可用（不重建索引） | PASS | 真实 E2E：持久化后模拟重启，`load_persisted_index` 返回 `loaded`，检索仍命中 `scanned.pdf#0`（P2.1 同项测试 `test_persisted_index_restores_after_restart`）。 |
| 一份 image-only PDF（纯图像 PDF，无文本层）完成入库并命中正确 chunk | PASS | 合成 image-only PDF → OCR 得 `HYDRAULIC PUMP PRESSURE`（末词 `LOW` 因合成图右侧边缘裁切未被识别）→ 入库 → 查询命中 `scanned.pdf#0`（向量 top1；BM25 命中见 §4）。 |
| 混合型 PDF（1 页扫描）只对扫描页 OCR | PASS | 见 P2.5。 |
| chunk 页码抽样验证通过 | PASS | 6/6。 |
| `pytest` 全绿 | PASS | 107 passed。 |
| `docs/baseline-defects.md` D4、D5 标记「已修复」 | PASS | 已更新 D4/D5 状态。 |

## 4. 端到端验证（合成 image-only PDF + 真实 RapidOCR，独立临时目录）

- 数据目录：系统临时目录，未触碰项目 `data/index/`，未写入样例 PDF。
- 扫描样本：**合成 image-only PDF（纯图像 PDF，无文本层）**，仅用于验证扫描件 OCR 流程，**非真实业务扫描件**。文本层 0 字符 → `is_scanned_document=True` → RapidOCR（离线）得到 `HYDRAULIC PUMP PRESSURE`（末词 `LOW` 因合成图右侧边缘裁切未被识别），`ocr_used=True`。
- 混合型：文本页 2661 有效字符 + 扫描页 → 整篇非扫描件（平均 1330 字符/页）→ 仅第 2 页 OCR。
- 入库 + 检索：1 图像页 + 2 文本页；查询 `HYDRAULIC PUMP PRESSURE` → 向量 top1 = `scanned.pdf#0`；BM25 亦命中该 chunk（注意：语料规模过小时 `rank_bm25` 的 IDF 可能非正，被 `bm25_index.search` 的 `score>0` 过滤，需足够语料规模才能命中）。
- 持久化恢复：`save_persisted_index` → 新实例 `load_persisted_index` = `loaded`；检索仍命中 `scanned.pdf#0`，恢复元数据 `{source=scanned.pdf, page=1, ocr_used=true, chunk_index=0}`。
- chunk 页码抽样 6/6 通过。

## 5. 测试与覆盖率

- `pytest -q -p no:cacheprovider`：**107 passed / 0 failed / 0 skipped / 1 warning**（warning 为 gradio 既有 `DeprecationWarning`）。
- `pytest -q --cov -p no:cacheprovider`：覆盖率 **78.29%**（阈值 60%）；`core/ocr.py` 93%、`core/text_splitter.py` 92%、`core/document_loader.py` 83%、`core/vector_store.py` 95%、`core/bm25_index.py` 82%。
- 较 P2.1 基线（79 passed / 74.58%）：新增 28 用例（P2.2 与 P2.3–P2.8）。

## 6. 依赖与配置

- 新增依赖（已写入 `requirements.txt`）：`rapidocr_onnxruntime>=1.4.0`（实际 1.4.4）、`PyMuPDF>=1.24.0`（实际 1.28.2）。二者均为纯 pip wheel，无需系统级软件。
- 配置沿用 `config.py` 既有 `OCR_ENABLED/OCR_ENGINE/OCR_RENDER_DPI/SCAN_TEXT_THRESHOLD/OCR_MAX_PAGES/INDEX_DIR`，未新增配置项。

## 7. 未验证项 / 已知局限

- 本阶段**无真实业务扫描件样本**；OCR 流程用**合成 image-only PDF（纯图像 PDF）**验证（清晰印刷体）。手写体与低质量扫描件识别不在范围（WBS §14）。
- OCR 识别质量：部分合成图会出现英文词间空格丢失（属 RapidOCR 识别特性，非本层代码缺陷）；中文场景不受影响。
- `OCR_MAX_PAGES=50` 仍为待确认默认值（WBS 未指定）。
- 端到端「完整问答生成」未执行（未接 LLM 生成）；P2 验证止于「入库 + 检索命中」。
- P1 遗留项（默认 `BAAI/bge-reranker-base` 真实加载、`RERANK_METHOD=llm`、`HYBRID_ALPHA` 校准、端到端延迟、远程 CI）状态不变；远程 CI 仍未验证。

## 8. 结论

- P2.1–P2.9 全部完成；P2.9 六项验收标准均 **PASS**；`docs/baseline-defects.md` 的 D4、D5 标记「已修复」。
- 已知局限见 §7，均为范围外或环境受限项，不阻塞 G2。
