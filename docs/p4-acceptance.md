# P4 前端重写 · 验收记录

> 依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》§7 P4（约 603–705 行）、§11.4 / §11.5 与 §13 G4。
> 记录原则：只记录**已验证**事实；未验证 / 受限项明确标注，不写成已通过。

## 1. 范围与依赖

- 任务：P4.1–P4.8。依赖 P3.5 流式事件协议（已冻结）、P2.1 索引持久化校验、P2.7 chunk 元数据。
- 环境：Python 3.11.7（`.venv`）；Windows；CPU（无 GPU 加速）；测试与运行均设 `HF_HUB_OFFLINE=1`。
- **LLM 后端限制**：本机无可用 Ollama，且未获授权调用云端付费 API。因此端到端验证中**回答文本使用确定性替身**
  （在运行的服务进程内替换 `core.generator.stream_answer`）；而检索、索引、SSE 管线、引用校验与前端渲染
  均走**真实执行路径**。见 §5、§7。
- 浏览器验证：Microsoft Edge 155（headless + DevTools Protocol），真实加载 `web/` 页面并驱动交互
  （填入问题、点击提问、点击引用 chip），不联网、不下载模型。

## 2. 改动 / 新增文件

- 新增 `web/index.html`、`web/app.css`、`web/app.js`：P4.1–P4.7 前端，原生手写，无框架、无构建链。
- 新增 `core/ingest.py`：P4.2 入库编排（`ingest_files` / `list_documents` / `ALLOWED_EXTENSIONS`），从 Gradio 入口迁出。
- 新增 `core/system_info.py`：P4.6 模型信息（自 `rag_demo.py` 原样迁出，去除 UI 依赖）。
- `api_router.py`：P4.1 `StaticFiles` 挂载 `/static` + `GET /`；P4.2 `POST /api/upload`（多文件 + `force_ocr`）与 `GET /api/documents`；
  P4.6 `/api/status` 扩展索引/模型字段。
- `requirements.txt`：移除 `gradio` 依赖（保留说明注释）。
- `rag_demo.py`：**删除**（P4.1）。
- 删除 `tests/test_gradio_history.py`（随 Gradio 入口移除）。
- `tests/test_system_models_info.py`：改为 `import core.system_info`（断言不变）。
- 新增 `tests/test_ingest.py`（7 条）、`tests/test_api_web.py`（8 条）。
- `README.md` / `README.zh-CN.md`：更新界面说明与启动方式，新增「界面预览」并引用新截图。
- 新增截图：`images/p4-01-overview.png`、`images/p4-02-chat-citations.png`、`images/p4-03-retrieval-panel.png`、`images/p4-04-dark-narrow.png`。

## 3. 逐任务验收

### P4.1 静态资源与路由骨架 — PASS

- 产出物齐全：`web/index.html` / `app.css` / `app.js`；`api_router.py` 挂载 `StaticFiles(directory=web)` 到 `/static`，并新增 `GET /` 返回 `web/index.html`。
- 无前端框架 / 无构建链：页面只引用 `/static/app.css`、`/static/app.js`。
  `tests/test_api_web.py::test_index_html_uses_no_frontend_framework` 断言 html 不含 `react`/`vue`/`angular`/`svelte`；
  `test_static_assets_served` 断言 `/static/app.css`（text/css）与 `/static/app.js`（javascript）均 200。
- Gradio 入口与依赖已移除：`rag_demo.py` 不存在；`requirements.txt` 非注释行不含 `gradio`。
  `test_gradio_entry_and_dependency_removed` 断言二者。
- 根路径返回页面：`test_index_page_served`（200，含「本地 RAG」「app.js」「app.css」）；E2E 中 `GET /` 返回完整页面。
- 证据：`tests/test_api_web.py`（4 条）、`images/p4-01-overview.png`。

### P4.2 上传与文档列表 — PASS

- 上传接口：`POST /api/upload` 支持多文件字段 `files`（并保留旧单文件字段 `file` 兼容）；表单参数 `force_ocr`；
  临时文件保留**原始文件名**，处理后清理。空请求返回 400。
- 逐文件结果 + OCR 判定：`core/ingest.py::ingest_files` 返回
  `{status,message,files[{filename,status,chunks,ocr_used,message}],total_chunks,persisted}`；单文件失败只记录该文件错误并继续其余文件。
- 文件类型白名单沿用既有：`{.pdf,.txt,.docx,.xlsx,.xls,.pptx,.md}`（`ALLOWED_EXTENSIONS`）。
- 文档列表：`GET /api/documents` → `{documents[{source,doc_id,chunks,ocr_used,page_count}],total_chunks,index_ready}`，数据取自真实 chunk 元数据。
- 上传进度可见：前端上传期间展示进度条（indeterminate）+「已用时 Ns」计时，并逐文件列出结果。
- 扫描件「已启用 OCR」标识（**真实端到端验证**）：上传 1 个合成的 image-only PDF
  （`scanned_pump_notice.pdf`，由 PyMuPDF 将文字渲染为位图后整页插入，页面文本层长度为 0）与 3 个文本文件，
  响应 `total_chunks=4`，其中 `scanned_pump_notice.pdf → status=success, chunks=1, ocr_used=true`；
  `GET /api/documents` 中该项 `ocr_used=true, page_count=1`；界面渲染为
  `scanned_pump_notice.pdf [已启用 OCR] · 1 chunks · 1 页`（见 `images/p4-01-overview.png`）。
  OCR 识别文本为 `HYDRAULICPUMPPRESSURELOWCHECKVALVE`（合成图边缘裁切导致空格丢失，与 P2 记录一致，不夸大识别质量）。
- 证据：`tests/test_ingest.py`、`tests/test_api_web.py::test_upload_returns_per_file_results` /
  `test_upload_supports_legacy_single_file_field` / `test_upload_without_files_returns_400` / `test_documents_endpoint_returns_list`；
  E2E 上传结果；`images/p4-01-overview.png`。

### P4.3 对话区（SSE 消费）— PASS

- 用 `fetch('/api/ask/stream')` + `ReadableStream` 手工解析 SSE 帧（跳过 `:` 心跳注释帧），逐片 `delta` 渲染（非一次性）。
- **首字延迟可见**：首个 `delta` 到达时记录客户端「首字延迟」；`done` 后状态栏显示
  `完成 · 首字延迟 X ms · 客户端总耗时 Y ms · 服务端耗时 Z ms`。
- **本轮纠偏（P4.3 相关缺陷）**：原实现中 `done` 写入的首字延迟文案会被 `finally` 块立即覆盖为「就绪」，
  导致该数字实际不可见，违反 P4.3「首字延迟可见（必须在界面上体现）」。修复方式：`done` 置 `turn.done=true`，
  `finally` 仅在未完成且无错误时回落为「就绪」；同时把原先混为一谈的「总耗时」拆成客户端 / 服务端两个口径，消除量纲混淆。
- 错误可读：SSE `error` 事件与 HTTP 失败均显示中文文案，不显示堆栈。
- 多轮展示：会话内保留最近 5 轮（`MAX_TURNS=5`），超出时移除最旧一轮 DOM。
- 证据：E2E 实测状态栏「首字延迟 22 ms」；`web/app.js`；`images/p4-02-chat-citations.png`。

### P4.4 引用可点定位 — PASS

- 正文 `[n]` 渲染为 `<span class="cite" data-index="n">`；点击后切换到对应轮次的检索面板并高亮
  `data-chunk-index=n` 的 chunk（`scrollIntoView`）。
- 映射契约：`citations[].index` → `chunks[].index`（两者均为请求内序号、与数组下标一致），不从编号反推 chunk_id（§1.1.4）。
- 来源与页码：引用 chip 显示 `[i] source · 第N页`，仅 `page>0` 时显示页码；chunk 卡片同规则。
- 校验失败：`citation_passed===false` 时在答案**上方**追加提示（列出 `unsupported_claims`），**不撤回**已输出的正文（§1.1.5）。
- 证据：E2E 点击 `[0]` 后 `#retrieval-list .chunk.highlight` 数量 = 1；`images/p4-03-retrieval-panel.png`。

### P4.5 检索过程面板 — PASS

- 数据来源仅为 `retrieval` 事件字段，前端**不计算任何分数**。
- 展示内容：候选数、`score_source`、降级状态、改写后 query 与命中规则、检索 trace；每个候选显示
  `dense_score` / `bm25_score` / `fused_score` /（重排成功时）`rerank_score` / `rank_before → rank_after`（升降箭头 `▲▼–`）
  / `preview` / `source` / 页码 / `[OCR]`。
- 生成过程中即可见：`retrieval` 事件先于 `delta` 发出，面板随事件到达立即渲染。
- 排名字段语义（本轮实测核实，未改代码）：
  - `rank_before` = 该 chunk 在 `hybrid_merge` 融合候选列表中的位次；
  - `rank_after` = 重排（`score_source="rerank"`）或 RRF 降级（`score_source="rrf"`）之后的位次。
  - 故 `rerank` 路径反映**重排前后**；`rrf` 路径反映**融合排序 → RRF 降级排序**。
    截图 `images/p4-02-chat-citations.png` 为 RRF 路径，可见 `rank 1 → 2 ▼`、`rank 2 → 1 ▲`。
- `fused_score` 语义沿用 `docs/p3-acceptance.md` §8.4：恒为 `hybrid_merge` 融合分数，与是否重排无关；
  RRF 降级只改排序、不写入 `fused_score`；重排成功时 `rerank_score` 为 CrossEncoder 分数，降级时为 `null`。
- 证据：E2E `retrieval` 载荷（见 §5）+ 截图。

### P4.6 索引与持久化状态 — PASS

- 展示：索引目录（`index_dir`）、已索引文档数（`indexed_documents`）、chunk 总数（`total_chunks`）、embedding 模型名、
  索引构建时间（`index_built_at`），以及 `models` 与 `model_choices` / `default_model`。
- 三态：由 `index_ready` + `index_status`（`loaded` / `not_ready` / `mismatch`）驱动顶部徽章与面板文案；
  本轮修正原先「灌库后仍显示 `not_ready`」的逻辑，使 `vector_store.is_ready` 时强制为 `loaded`。
  E2E 中灌库后 `GET /api/status` 返回 `index_status="loaded"`、`indexed_documents=4`、`total_chunks=4`。
- 可操作修复提示：`mismatch` → 「索引参数与当前配置不一致，请重新上传文档以重建索引」；
  `not_ready` → 「尚未建立索引，请先上传并处理文档」。
- 证据：`tests/test_api_web.py::test_status_exposes_index_state_and_models`；E2E `/api/status`；截图中「索引与模型」面板。
- 未验证：`mismatch`（参数不一致）分支本轮未构造真实数据触发，仅代码实现 + 前端映射，标记 **NOT VERIFIED（真实触发）**。

### P4.7 主题与响应式 — PASS

- 浅色为主 + 深色切换：CSS 变量切换由 `body[data-theme="dark"]` 控制，偏好写入 `localStorage["rag_theme"]`，页面加载时读取。
- 窄屏降级：`@media (max-width:1024px)` 将三栏改为纵向堆叠。
- 实测：1440px 三栏（p4-01/02/03）与 768px 单栏堆叠 + 深色（p4-04）均正常渲染；CDP 设置 `innerWidth=768`、
  `document.body.dataset.theme="dark"` 后截图确认。
- 证据：`images/p4-01-overview.png`、`images/p4-04-dark-narrow.png`、`web/app.css`。

### P4.8 P4 验收 — PASS

- 端到端（浏览器内）：上传（含扫描件 OCR）→ 流式问答 → 点引用定位 → 检索过程面板，全流程无阻塞（见 §5）。
- 首字延迟实测数字：替身 LLM 下 **首字延迟 22 ms**（客户端口径，多次运行 22–24 ms）；真实 LLM 首字延迟**未验证**（无可用后端）。
- 无前端框架依赖：见 P4.1。
- README 引用新界面截图：`README.md`「Interface preview」与 `README.zh-CN.md`「界面预览」引用上述 4 张截图（≥3 张）。
- 证据：`images/p4-01-overview.png` … `p4-04-dark-narrow.png`；`README.md` / `README.zh-CN.md`。

## 4. 接口契约验收（§11.5）

| 接口 | 方法 | 本轮验证 | 结果 |
| --- | --- | --- | --- |
| `/` | GET | `test_index_page_served` + E2E 导航 | 200，返回单页 |
| `/static/*` | GET | `test_static_assets_served` + E2E 渲染 | 200，css/js 正确 MIME |
| `/api/upload` | POST | `test_upload_*` + E2E 多文件真实上传 | 200，逐文件结果 + `force_ocr` 生效 |
| `/api/documents` | GET | `test_documents_endpoint_returns_list` + E2E | 200，真实元数据 |
| `/api/status` | GET | `test_status_exposes_index_state_and_models` + E2E | 200，索引三态 + 模型字段 |
| `/api/ask` | POST | 既有测试（`tests/test_api_status.py` 等）保持通过，未改签名 | 兼容 |
| `/api/ask/stream` | POST | E2E 真实 SSE 探测 | 200，事件顺序 `retrieval → delta* → citation → done` |

## 5. 端到端验证（真实 HTTP：uvicorn + 真实索引 + 确定性替身 LLM）

- 启动（PowerShell，测试实例，端口 17996；进程内替换 `core.generator.stream_answer` 为确定性替身；`INDEX_DIR` 指向临时目录）：

```powershell
$env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -m uvicorn api_router:app --host 127.0.0.1 --port 17996
```

- 上传：`POST /api/upload`，4 个文件（3 个文本 + 1 个合成 image-only PDF）。
  结果：`total_chunks=4`，`persisted=true`；`scanned_pump_notice.pdf` 为 `ocr_used=true`。
- `GET /api/documents`：4 条文档；`scanned_pump_notice.pdf` → `ocr_used=true, page_count=1`。
- `GET /api/status`：`index_status="loaded"`、`indexed_documents=4`、`total_chunks=4`、`embedding_model="all-MiniLM-L6-v2"`。
- `POST /api/ask/stream`（真实 SSE）实测事件顺序：`['retrieval', 'delta', 'delta', 'delta', 'citation', 'done']`。
  - `retrieval.chunks[0]` = `{index:0, source:"troubleshooting_guide.md", page:0, chunk_index:0,
    dense_score:0.4731, bm25_score:0.4107, fused_score:0.4544, rerank_score:null, rank_before:2, rank_after:1, ocr_used:false}`。
  - `citation` 载荷 = `{"citations":[{"index":0,"source":"troubleshooting_guide.md","page":0,"chunk_index":0}],
    "citation_passed":true,"unsupported_claims":[]}` —— `citations[].index` 与 `chunks[].index` 一致。
  - `done` 载荷 = `{"session_id":"accept-p4","latency_ms":17,"token_usage":null}`。
- 浏览器（Edge headless + CDP）：导航 `/` → 填入问题 → 点击提问 → 等待完成 → 点击引用 chip → 截图。
  状态栏 `首字延迟 22 ms · 客户端总耗时 25 ms · 服务端耗时 20 ms`；点击后高亮 chunk 数 = 1。

## 6. 产出截图

| 文件 | 内容 |
| --- | --- |
| `images/p4-01-overview.png` | 上传区、已入库文档（含扫描件「已启用 OCR」标识）、索引与模型面板 |
| `images/p4-02-chat-citations.png` | 流式问答、引用 chip、检索过程面板（含 rank_before → rank_after 升降） |
| `images/p4-03-retrieval-panel.png` | 点击引用后对应 chunk 被高亮 |
| `images/p4-04-dark-narrow.png` | 深色主题 + 768px 窄屏纵向堆叠 |

> 截图中的回答文本来自确定性替身 LLM；检索候选、分数、`score_source`、引用定位、索引状态均为真实数据。

## 7. 未验证项 / 已知局限

1. **真实 LLM**：云端 `stream=True` 运行时、真实改写/摘要质量、`token_usage` 解析（当前恒为 `null`）均未验证（无授权后端）。替身 LLM 下的首字延迟数字（22 ms）**不能**代表真实模型延迟。
2. **默认重排模型**：`BAAI/bge-reranker-base` 未在本环境完成真实加载验证；截图中的 `重排服务故障 → 回退 RRF` 即因此产生。
3. **`RERANK_METHOD=llm`** 未真实验证。
4. **`HYBRID_ALPHA` 校准**留待 P5；当前默认 0.7 未做效果调优。
5. **`index_status="mismatch"`** 分支未构造真实数据触发（仅实现 + 前端映射）。
6. **远程 CI** 未验证；P0 遗留的端到端真实问答耗时仍受环境限制。
7. 现有 P2/P3/P4 改动**尚未提交**（工作区保留），未创建 commit、未推送。

## 8. G4 结论

G4 要求：浏览器内可完成「上传 → 流式问答 → 点引用定位 → 看到检索过程」。

- 上传（含扫描件 OCR）：PASS（真实 API + 界面标识）。
- 流式问答（逐片渲染 + 首字延迟可见）：PASS（真实 SSE）。
- 点引用定位：PASS（引用 chip → chunk 高亮）。
- 检索过程面板：PASS（真实 `retrieval` 字段）。
- 无前端框架：PASS。

**结论：G4 SATISFIED**（以上 5 项均有本轮实测证据）。
限制：真实 LLM 与默认重排模型的运行时验证仍为未验证项（见 §7），不阻塞 G4 的功能验收，但需在 P6/P7 交付前解决或如实披露。
