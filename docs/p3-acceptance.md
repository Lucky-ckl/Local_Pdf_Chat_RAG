# P3 会话与生成链 · 验收记录

> 依据《Local_Pdf_Chat_RAG_WBS技术执行文档 v1.0》§6 P3（约 505–599 行）与 §11.3/§11.4、§1.1.5。
> 记录原则：只记录**已验证**事实；未验证/受限项明确标注，不写成已通过。

## 1. 范围与依赖

- 任务：P3.1–P3.7；依赖 P2.7（chunk 元数据）、P0.3（`SESSION_MAX_TURNS` / `SSE_HEARTBEAT_SEC`）。
- 环境：Python 3.11.7（`.venv`）；Windows；CPU（无 GPU）。
- **LLM 后端限制**：本机无可用 Ollama，且未获授权调用云端付费 API。因此**真实 LLM 调用未执行**；
  改写/摘要/生成环节以**确定性替身**（注入 `rewrite_fn`、monkeypatch `call_llm_simple` / `stream_answer`）验证机制，
  并经**真实 HTTP（uvicorn + curl）**验证 SSE 管线。见 §4、§6。

## 2. 改动 / 新增文件

- `core/session.py`（新增）：P3.1 内存会话状态（`Turn` / `Session` / `SessionStore`，有界轮次）。
- `core/retriever.py`：P3.2 `RewriteResult` + `rewrite_query_with_history`；`recursive_retrieval(..., history=, rewrite_out=, trace_out=)`；
  `hybrid_merge` 载荷补充 `dense_score` / `bm25_score`；召回候选 trace（含 rerank 前后排名、`score_source`）。
- `core/generator.py`：P3.2 `query_answer` / `stream_answer` 接收 `history`；P3.3 `compress_history`（摘要缓存 + 长度受控）；
  P3.4 `_stream_openai_compatible_api`（真实 `stream=True`）/ `_stream_llm` / `stream_answer`（增量主路径）/ `stream_answer_events`（事件序列）；
  P3.5 `verify_citations`；`_build_prompt(..., history_block=)`。
- `api_router.py`：P3.4–P3.6 `POST /api/ask/stream`（SSE）+ `_sse_event_stream`（心跳 / 断连 / 错误）；`/api/ask` 支持 `session_id` 多轮。
- 测试：`tests/test_session.py`、`tests/test_query_rewrite.py`、`tests/test_prompt_compression.py`、`tests/test_sse_events.py`、`tests/test_multiturn_rewrite_hits.py`。
- `docs/baseline-defects.md`：D2、D3 标记「已修复」。
- `rag_demo.py`（P3 多轮入口）：新增 `_gradio_history_to_turns`，把 Gradio messages 历史转为 `(question, answer)` 轮次并传入 `query_answer(..., history=...)`，修复「Gradio 历史从未传入」（D2 的 UI 入口）。
- `core/generator.py::is_llm_failure`：识别 LLM 错误占位串（未配置 Key / HTTP 错误等），供查询改写判定「改写失败」并回退原 query。
- 测试（协议缺口修复轮新增/扩展）：`tests/test_gradio_history.py`（新增）、`tests/test_sse_events.py`、`tests/test_rerank_contract.py`、`tests/test_query_rewrite.py`。

## 3. 逐任务验收

### P3.1 会话状态结构

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 会话可创建 / 读取 | PASS | `core/session.py::SessionStore`；`test_session_create_read_and_truncate_drops_oldest`、`test_get_or_create_reuses_existing_session`。 |
| 超出轮数截断，最旧轮次被丢弃 | PASS | `test_session_create_read_and_truncate_drops_oldest`：5 轮写入 → 仅保留 3 轮 `[q2,q3,q4]`。 |
| 显式说明内存设计边界 | PASS | `core/session.py` 模块 docstring 声明「仅存内存 / 不引入 Redis」（WBS §13）。 |

### P3.2 history 进查询改写

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 省略追问被改写为自包含问题并命中正确 chunk | PASS | `tests/test_multiturn_rewrite_hits.py`：3 组样本（保修期 / 保养周期 / 价格），真实 BM25 检索，`rewrite_used=True` 且命中期望 chunk。 |
| 无历史时不改写 | PASS | `test_no_history_does_not_rewrite`：`rewrite_used=False` 且改写函数 0 次调用。 |
| 改写失败不阻塞检索 | PASS | `test_rewrite_failure_falls_back_to_original`、`test_empty_or_overlong_rewrite_falls_back`：异常/空/超长均回退原 query。 |
| 返回 `rewritten` / `rewrite_used` / `rewrite_rule` | PASS | `RewriteResult`；`test_history_triggers_rewrite_with_rule`、`test_recursive_retrieval_uses_rewritten_query`。 |

### P3.3 多轮上下文压缩

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| prompt 长度不随轮次线性增长 | PASS | `compress_history(..., max_chars=...)`；`test_prompt_length_bounded_regardless_of_turn_count`（10 轮，长度 ≤ 上限）。 |
| 摘要生成有缓存 | PASS | `test_summary_generated_once_and_cached`：同一批较早轮次重复调用，LLM 仅 1 次。 |
| 最近 1 轮完整、更早轮压缩 | PASS | `test_compress_history_single_turn_keeps_full_qa`、`test_build_prompt_includes_history_block`。 |
| 摘要失败降级不报错 | PASS | `test_summary_failure_falls_back_to_bounded_truncation`。 |

### P3.4 SSE 流式生成接口

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| `curl -N` 能看到逐片输出 | PASS | 真实 uvicorn + `curl -N POST /api/ask/stream`：依次收到 `retrieval` → 3×`delta` → `citation` → `done`（见 §4）。 |
| 长回答期间有心跳 | PASS | `test_sse_emits_heartbeat_when_generation_stalls`：停顿超过 `SSE_HEARTBEAT_SEC` 时下发 `: heartbeat` 帧。 |
| 非流式 `/api/ask` 仍可用 | PASS | `curl /api/status` 正常；`/api/ask` 路由保留（`api_router.py`）。 |
| 修掉云端写死 `"stream": False` | PASS | `_stream_openai_compatible_api` 以 `stream=True` 请求；`stream_answer` 逐片 `yield`。 |

### P3.5 流式事件协议

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 事件顺序 retrieval → delta → citation → done | PASS | `test_event_order_and_fields`；`curl` 实测顺序一致（§4）。 |
| `index` 与 `chunks` 下标一致 | PASS | `recursive_retrieval` trace 以 `enumerate(reranked)` 生成 `index=pos`；`test_event_order_and_fields` 断言 `index==0`。 |
| `citation` 事件含 `{citations, citation_passed, unsupported_claims}`（§11.4） | PASS | `verify_citations` 回填 `citations:[{index,source,page,chunk_index}]`（仅含 `index` 存在且 `source/page/chunk_index` 键齐全的可定位引用；`index` 与候选下标一致）；测试：`test_verify_citations_returns_resolved_citations_and_flags_out_of_range`、`test_verify_citations_multiple_references_keep_index_mapping`、`test_verify_citations_missing_metadata_is_not_locatable`、`test_verify_citations_partial_metadata_is_not_locatable`、`test_verify_citations_zero_page_and_chunk_index_are_valid`、`test_verify_citations_mixed_locatable_and_missing_metadata`、`test_verify_citations_duplicate_reference_is_deduplicated`、`test_verify_citations_empty_chunks_flags_reference`、`test_verify_citations_without_reference_returns_empty_citations`。 |
| `retrieval` 事件含降级状态与 trace（§11.3） | PASS | 载荷新增 `rerank_disabled_by_user`/`rerank_unavailable`/`retrieval_trace`，`RankedChunk` 补 `fused_score`/`ocr_used`/`chunk_index`；`test_retrieval_event_surfaces_rerank_disable_and_failure`、`test_event_order_and_fields`；`tests/test_rerank_contract.py` 的 `test_trace_reports_rerank_success`、`test_trace_distinguishes_user_disabled_from_service_failure`、`test_trace_chunk_exposes_metadata_and_fused_score`、`test_trace_chunk_metadata_defaults_when_missing`、`test_trace_reports_only_executed_stages`。 |

### P3.6 断连与错误处理

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 客户端断连 → 终止 LLM 调用（日志证据） | PASS | 真实 uvicorn + 硬关闭（RST）客户端：服务端日志 `SSE 流提前终止…已关闭生成器（session=rst-sess）`；单测 `test_sse_disconnect_closes_generator_and_logs` 断言 `close()` 与日志。 |
| 检索失败 → 发 `error`，不发空 delta | PASS | `test_retrieval_failure_emits_error_without_delta`。 |
| 生成中途失败 → 已输出正文保留 + `error` | PASS | `test_generation_failure_keeps_partial_and_appends_error`。 |

### P3.7 P3 验收

| 验收标准 | 结果 | 证据 |
| --- | --- | --- |
| 多轮省略追问命中正确 chunk（≥3 组） | PASS | `tests/test_multiturn_rewrite_hits.py`：3 组通过。 |
| `curl -N` 的 SSE 输出完整且顺序正确 | PASS | §4。 |
| 断连后服务端有终止日志 | PASS | §4。 |
| `pytest` 全绿 | PASS | §5。 |
| `docs/baseline-defects.md` D2、D3 标记「已修复」 | PASS | 已更新 D2/D3。 |

## 4. 端到端验证（真实 HTTP：uvicorn + curl / RST 客户端）

- 方式：本机启动 uvicorn（`api_router.app`），LLM 以**确定性替身**注入（不联网、不调用付费 API）；
  `curl -N` 走真实 HTTP 与 SSE。
- 正常流（`curl -N -X POST /api/ask/stream`，`session_id=curl-sess`）实际输出：

  ```text
  event: retrieval   data: {chunks:[{index:0,...}], score_source:"rerank", rewritten:"液压泵压力低怎么办", rewrite_used:false}
  event: delta       data: {"text":"液压泵"}
  event: delta       data: {"text":"压力过低"}
  event: delta       data: {"text":" [0]。"}
  event: citation    data: {"citation_passed":true,"unsupported_claims":[]}
  event: done        data: {"session_id":"curl-sess","latency_ms":1208,"token_usage":null}
  ```

- 断连流：客户端读若干 delta 后以 `SO_LINGER=0` 硬关闭（RST）；服务端日志出现
  `INFO:rag-api:SSE 流提前终止（客户端断连或异常中断），已关闭生成器（session=rst-sess）`。
- 非流式兼容：`GET /api/status` 返回 200 正常 JSON。
- 说明：`token_usage` 暂为 `null`（未解析云端流式 usage），非阻塞项。
- 说明（协议缺口修复轮）：本段为**修复前**的 curl 实测输出，当时 `citation` 事件缺 `citations` 字段；
  修复后的完整事件以 §8（本轮离线 in-process 复核）为准。

## 5. 测试与覆盖率

- 首版 P3 验收：`pytest -q -p no:cacheprovider`（离线）→ **136 passed / 0 failed / 0 skipped / 1 warning**；`--cov` → **78.69%**。
- **协议缺口修复轮**：`pytest -q -p no:cacheprovider`（离线，`HF_HUB_OFFLINE=1`）→
  **152 passed / 0 failed / 0 skipped / 1 warning**（gradio 既有 `DeprecationWarning`）；
  `pytest -q --cov -p no:cacheprovider` → 覆盖率 **80.04%**（阈值 60%）：
  `core/session.py` 100%、`core/retriever.py` 89%、`core/generator.py` 49%（未覆盖多为云端/LLM 网络分支）、`core/vector_store.py` 95%。
- P3 新增用例（首版）：`test_session.py`(5) + `test_query_rewrite.py`(6) + `test_prompt_compression.py`(7) + `test_sse_events.py`(8) + `test_multiturn_rewrite_hits.py`(3) = 29；基线 107 → 136。
- 本轮新增：`test_sse_events.py` +5（多引用/空引用/缺元数据/越界引用/事件降级状态）、`test_rerank_contract.py` +5（retrieval trace 契约）、`test_query_rewrite.py` +2（错误串判定与回退）、`tests/test_gradio_history.py` +4；136 → 152。
- **P3 最终收尾轮（本轮）**：`pytest -q -p no:cacheprovider`（离线，`HF_HUB_OFFLINE=1`）→
  **156 passed / 0 failed / 0 skipped / 1 warning**（同上 `DeprecationWarning`）；
  `pytest -q --cov -p no:cacheprovider` → 覆盖率 **80.13%**（阈值 60%）：`core/session.py` 100%、`core/retriever.py` 89%、`core/generator.py` 50%、`core/vector_store.py` 95%。
- 本轮用例变化：`tests/test_sse_events.py` 将原「缺元数据取默认值」1 例改写为引用定位边界 4 例（缺元数据 / 部分缺元数据 / `page=0`·`chunk_index=0` 合法 / 混合可定位），净 +3；未新增/删除其他阶段测试。

## 6. 未验证项 / 已知局限

- **真实 LLM 调用未执行**：无可用 Ollama，且未获授权调用云端付费 API。改写/摘要/生成以确定性替身验证**机制**；
  真实模型下的改写质量与流式首字延迟**未验证**（留待有后端时补测）。
- `token_usage` 未解析（恒 `null`）。
- 会话为**进程内内存**：多 worker / 重启后丢失（WBS §13 明确本期不实现分布式会话）。
- Gradio 多轮入口本轮已补 `history` 传递（`rag_demo.py::_gradio_history_to_turns`）；其真实多轮改写质量仍依赖可用 LLM 后端，未执行。
- 查询改写对「LLM 返回错误占位串」按失败处理（`is_llm_failure`，基于错误前缀的**启发式**）并回退原 query；非严格错误码校验。
- P1 遗留项状态不变（默认 `BAAI/bge-reranker-base` 真实加载、`RERANK_METHOD=llm`、`HYBRID_ALPHA` 校准、端到端延迟、远程 CI）；P0 遗留 `OCR_MAX_PAGES=50` 仍待确认。

## 7. 结论

- P3.1–P3.7 全部实现；P3.7 五项验收标准均 **PASS**（含 3 组多轮命中、curl SSE、断连终止日志、全绿、D2/D3 已修复）。
- 已知局限见 §6：真实 LLM 端到端未执行，属**环境/授权受限**，机制已用替身 + 真实 HTTP 验证，不阻塞 G3。

## 8. 协议缺口修复记录（P3 收尾，本轮）

> 背景：P3 只读验收审查发现 P3.5 的两处契约缺口。本节记录修复内容与复核证据，不改变 P3 各任务的原始验收结论。

### 8.1 修复项

1. **`citation` 事件缺 `citations` 数组**（§6 P3.5 / §11.4）：
   `core/generator.py::verify_citations` 现返回 `{"citations": [{index, source, page, chunk_index}], "citation_passed", "unsupported_claims"}`；
   `citations` 仅含**存在且可定位**的引用（`index` 与 `retrieval.chunks` 数组下标一致，且 §11.2 定位元数据 `source/page/chunk_index` **键齐全**），按 `index` 升序；无有效引用时为空数组。按**键是否存在**判定，故 `page=0`（非 PDF 契约值）与 `chunk_index=0`（首块）等合法值不会被误判为缺失；元数据不全的编号**不回填、不虚构默认值**，计入 `unsupported_claims`（P3 收尾轮据 §11.2/§11.4 收紧，见 §8.4）。
2. **`retrieval` 事件缺 §11.3 字段**：
   `core/retriever.py::recursive_retrieval` 的 trace 现补充 `rerank_disabled_by_user`、`rerank_unavailable`、`retrieval_trace`；
   `RankedChunk` 补 `fused_score`（真实混合融合分数，取自 `hybrid_merge` 的 `score`）、`ocr_used`（真实 chunk 元数据，缺失为 `False`）、`chunk_index`（真实元数据，供 citation 定位）。
   降级状态取自**请求级** `RerankStatus`（`last_status`），用户关闭与服务故障可区分；`retrieval_trace` 只记录**实际执行**的阶段与耗时（`stages_ms`），未执行的步骤不写入。
3. **Gradio 多轮入口**：`rag_demo.py` 新增 `_gradio_history_to_turns` 并在 `process_chat` 传入 `query_answer(..., history=...)`。
4. **改写失败判定**：`core/generator.py::is_llm_failure` + `core/retriever.py::_default_rewrite_fn` 在 LLM 返回错误占位串/空结果时按「改写失败」回退原 query（避免把错误串当检索查询）。

### 8.2 本轮离线复核（in-process，真实检索 + 真实 SSE 包装，LLM 用确定性替身；不联网、不加载模型）

- 事件顺序实测：`retrieval → delta → delta → citation → done`。
- `retrieval.chunks[0]`：`{index:0, chunk_id:"hyd", source:"manual.pdf", page:3, chunk_index:7, dense_score:0.0, bm25_score:1.0, fused_score:0.3, rerank_score:0.9, rank_before:1, rank_after:1, ocr_used:false}`。
- `retrieval` 元字段：`score_source:"rerank"`、`rerank_disabled_by_user:false`、`rerank_unavailable:false`、`rewrite_used:true`、`rewrite_rule:"history_coreference"`、`retrieval_trace:{iterations:1, web_search_used:false, stages_ms:{rewrite,semantic,bm25,hybrid_merge,rerank}}`。
- `citation`：`{"citations":[{"index":0,"source":"manual.pdf","page":3,"chunk_index":7}],"citation_passed":true,"unsupported_claims":[]}`。
- SSE 包装输出 5 帧，首帧 `event: retrieval`、末帧 `event: done`；会话轮次 `citations` 已写回。

### 8.3 未验证项（与 §6 一致）

- 真实云端 `stream=True` 运行时、真实 LLM 改写/摘要质量、`token_usage` 解析、默认 reranker 真实加载、远程 CI —— 均未执行，属环境/授权受限。

### 8.4 `fused_score` 语义（P3 收尾轮核实，未改代码）

- `retrieval.chunks[].fused_score` **恒为** `core/retriever.py::hybrid_merge` 的融合分数
  （`alpha·dense_score + (1-alpha)·bm25_score`；两路均归一到 `[0,1]`，`HYBRID_ALPHA` 默认 0.7），
  与该次请求是否经过重排、是否回退 RRF **无关**（取值来自 `hybrid` 载荷，而非 `reranked`）。
- RRF 降级（`_rrf_fallback_ranking`）只改变**排序**；其 RRF 分数仅用于排序，**不写入** `fused_score`。
  因此 `fused_score` 在任何 `score_source` 下含义一致，不存在「因路径而异」的语义漂移。
- 分数通道由其他字段区分：`score_source`（`"rerank"` / `"rrf"`）、`rerank_disabled_by_user`、`rerank_unavailable`；
  重排成功时 `rerank_score` 为 CrossEncoder 分数，降级时为 `null`。P4 消费方须据此解释，不得把 `fused_score` 误当 CrossEncoder 分数。
- 结论：语义已准确，本轮**未改代码**；本说明供 P4.5「检索过程面板」参照。
