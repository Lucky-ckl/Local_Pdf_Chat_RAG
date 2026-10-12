"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import re
import json
from typing import Dict, Any, List, Optional
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from version import __version__

# 从重构后的模块导入
from config import (
    SILICONFLOW_API_KEY, MAGICK_API_KEY, is_configured_api_key, SSE_HEARTBEAT_SEC,
    INDEX_DIR, DEFAULT_MODEL_CHOICE, MODEL_CHOICES, MODEL_DISPLAY_NAMES,
)
from core.generator import query_answer, stream_answer_events
from core.ingest import list_documents, sync_documents
from core.session import session_store
from core.system_info import get_system_models_info
from core.vector_store import vector_store, load_persisted_index
from core.embeddings import EMBED_MODEL_NAME
from features.web_search import check_serpapi_key
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")
logger.setLevel(logging.INFO)  # 显式设置，避免根 logger 级别（可能为 WARNING）过滤掉 INFO 级断连/终止日志


INDEX_STATE = {"status": "unknown", "message": "", "chunks": 0}

# SSE 单流最大空闲等待（秒）：超过该时长仍无新事件则下发 error 并结束连接，避免无限心跳
# （P3.4 卡死修复，作为底层 LLM 空闲超时之外的兜底）。可用环境变量覆盖。
SSE_MAX_WAIT_SEC = float(os.getenv("SSE_MAX_WAIT_SEC", "120"))


# 后台知识库同步任务：持有强引用避免被垃圾回收，关闭时用于取消。
_background_tasks: set = set()


async def _run_kb_sync():
    """后台执行知识库增量同步：失败只记日志，不影响 API 启动与后续运行。"""
    try:
        summary = await asyncio.to_thread(sync_documents)
        logger.info("知识库同步: %s", summary.get("message"))
    except asyncio.CancelledError:
        logger.info("知识库同步任务已取消（服务关闭）")
        raise
    except Exception as e:
        logger.error("后台同步知识库失败: %s", e, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 服务启动")
    # P2.1：启动时加载已持久化的索引（无需重新灌库）
    global INDEX_STATE
    INDEX_STATE = load_persisted_index()
    logger.info("索引状态: %s - %s", INDEX_STATE.get("status"), INDEX_STATE.get("message"))
    # 知识库目录同步：扫描 data/documents，把新增文档增量入库（不清空已有文档）。
    # 改为后台任务：API 先启动完成（立即打印 Application startup complete），
    # 大量 PDF/OCR 的解析与入库在后台线程进行，不再阻塞启动。
    task = asyncio.create_task(_run_kb_sync())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    yield
    # 关闭时取消仍在进行的同步任务，不等待其完成（OCR 计算线程无法中断，仅停止等待）。
    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("API 服务已关闭")


app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型、云端模型服务和SERPAPI的文档问答API接口",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# P4.1：托管原生单页（无前端框架 / 无构建链）。静态资源由 FastAPI 直接挂载。
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
async def web_index():
    """返回前端单页（P4.1）。"""
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False
    model_choice: Optional[str] = None
    session_id: Optional[str] = None


class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@app.post("/api/upload")
async def upload_files():
    """已废弃（P6.1）：本版本统一采用知识库目录同步模式，不再接受上传。

    保留该路由仅为了让旧的调用方收到**明确错误**（而不是静默失败）。它**绝不**再调用
    ``ingest_files(reset=True)``，因此不会清空或覆盖已有索引。如需新增文档，请把文件放入
    知识库目录（``config.DOCUMENTS_DIR``，默认 ``data/documents``）后调用 ``POST /api/kb/refresh``。
    """
    raise HTTPException(
        status_code=410,
        detail="当前版本采用知识库目录同步模式，请将文件放入 data/documents 后点击同步",
    )


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """问答接口（非流式，兼容入口；P3.2：支持 session_id 多轮历史）"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    try:
        session = session_store.get_or_create(req.session_id)
        history = session.history()
        model_choice = req.model_choice or DEFAULT_MODEL_CHOICE
        answer = await asyncio.to_thread(
            query_answer, req.question, req.enable_web_search, model_choice, None, history
        )
        session_store.append_turn(session.session_id, req.question, answer, [])
        sources = []
        url_matches = re.findall(r'\[(网络来源|本地文档):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?', answer)
        for source_type, url in url_matches:
            sources.append({"type": source_type, "url": url} if url else {"type": source_type})

        return {
            "answer": answer, "sources": sources,
            "metadata": {"enable_web_search": req.enable_web_search, "model": model_choice,
                         "session_id": session.session_id}
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e


class AskStreamRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    enable_web_search: bool = False
    model_choice: Optional[str] = None


def _format_sse(event):
    """把事件字典序列化为 SSE 帧（P3.5）。"""
    return f"event: {event['event']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


async def _sse_event_stream(request, payload):
    """把 ``stream_answer_events`` 包装为带心跳与断连感知的 SSE 异步生成器（P3.4 / P3.6）。

    - 逐条转发事件；两帧之间超过 ``SSE_HEARTBEAT_SEC`` 未产出时发送心跳注释帧；
    - 客户端断连 → 记录日志并 ``close()`` 生成器（终止底层 LLM 流式调用，避免白付 token）；
    - 正常完成（收到 ``done``）后把本轮写入会话历史（供 P3.2 / P3.3 使用）。
    """
    session = session_store.get_or_create(payload.session_id)
    session_id = session.session_id
    history = session.history()

    generator = stream_answer_events(
        payload.question, session_id=session_id, history=history,
        enable_web_search=payload.enable_web_search,
        model_choice=payload.model_choice or DEFAULT_MODEL_CHOICE,
    )
    retrieval_chunks = []
    answer_parts = []
    completed = False
    disconnected = False
    pending = None
    last_event_at = time.monotonic()
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(asyncio.to_thread(_next_or_none, generator))
            done, _ = await asyncio.wait({pending}, timeout=SSE_HEARTBEAT_SEC)
            if not done:
                if time.monotonic() - last_event_at > SSE_MAX_WAIT_SEC:
                    logger.error("SSE 流空闲超过 %.0fs，主动结束（session=%s）", SSE_MAX_WAIT_SEC, session_id)
                    yield _format_sse({"event": "error", "data": {
                        "code": "generation_timeout",
                        "message": f"生成超时：超过 {SSE_MAX_WAIT_SEC:.0f} 秒未收到回答数据。",
                    }})
                    break
                yield ": heartbeat\n\n"
                continue
            event = pending.result()
            pending = None
            last_event_at = time.monotonic()
            if event is None:
                break
            name = event["event"]
            if name == "retrieval":
                retrieval_chunks = event["data"].get("chunks", []) or []
            elif name == "delta":
                answer_parts.append(event["data"].get("text", ""))
            elif name == "done":
                completed = True
            yield _format_sse(event)

            if await request.is_disconnected():
                disconnected = True
                logger.info("客户端断连，终止 LLM 调用（session=%s）", session_id)
                generator.close()
                return
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        try:
            generator.close()
        except Exception as exc:
            logger.warning("关闭 SSE 生成器失败（session=%s）：%s", session_id, exc)
        if not completed and not disconnected:
            # 客户端中断 / 异常导致流提前结束：关闭生成器即终止底层 LLM 流式调用，避免白付 token。
            logger.info("SSE 流提前终止（客户端断连或异常中断），已关闭生成器（session=%s）", session_id)

    if completed:
        answer_text = "".join(answer_parts)
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer_text)}
        citations = [c for c in retrieval_chunks if c.get("index") in cited]
        session_store.append_turn(session_id, payload.question, answer_text, citations)


@app.post("/api/ask/stream")
async def ask_stream(req: AskStreamRequest, request: Request):
    """流式问答接口（SSE，P3.4）：事件顺序 ``retrieval`` → ``delta``* → ``citation`` → ``done``。"""
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    return StreamingResponse(
        _sse_event_stream(request, req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _read_index_manifest():
    """读取索引 manifest（P2.1）用于展示构建时间 / 参数；缺失或损坏时返回 {}（不抛错）。"""
    path = os.path.join(INDEX_DIR, "manifest.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


@app.get("/api/documents")
async def get_documents():
    """已入库文档列表与处理结果（§11.5 ``GET /api/documents``，P4.2）。"""
    return {
        "documents": list_documents(),
        "total_chunks": vector_store.total_chunks,
        "index_ready": vector_store.is_ready,
    }


@app.post("/api/kb/refresh")
async def refresh_knowledge_base():
    """扫描知识库目录（``config.DOCUMENTS_DIR``）并增量入库新增文档。

    前端不再提供上传入口：用户把 PDF 放入知识库目录后，服务启动时或调用本接口即自动入库，
    已有文档不会被清空或覆盖。返回 ``sync_documents`` 的逐文件结果与汇总。
    """
    try:
        return await asyncio.to_thread(sync_documents)
    except Exception as e:
        logger.error("知识库同步失败: %s", e)
        raise HTTPException(500, f"知识库同步失败: {e}") from e


@app.get("/api/status")
async def check_status():
    """索引三态 + 模型信息（§11.5，P4.6）：所有取值来自真实运行态 / 真实 config。"""
    manifest = _read_index_manifest()
    # 索引三态以**当前运行态**为准：启动时加载成功后又被灌库（或本就加载成功）都应为 loaded。
    index_status = INDEX_STATE.get("status")
    index_message = INDEX_STATE.get("message")
    if vector_store.is_ready:
        index_status = "loaded"
        if INDEX_STATE.get("status") != "loaded":
            index_message = f"索引已就绪，共 {vector_store.total_chunks} 个文本块"
    return {
        "status": "healthy",
        "siliconflow_configured": is_configured_api_key(SILICONFLOW_API_KEY),
        "magick_configured": is_configured_api_key(MAGICK_API_KEY),
        "serpapi_configured": check_serpapi_key(),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks,
        "index_ready": vector_store.is_ready,
        "index_status": index_status,
        "index_message": index_message,
        # P4.6：索引文件路径 / 已索引文档数 / chunk 总数 / embedding 模型名 / 构建时间。
        "index_dir": INDEX_DIR,
        "index_built_at": manifest.get("built_at"),
        "index_embedding_model": manifest.get("embedding_model"),
        "indexed_documents": len(list_documents()) if vector_store.is_ready else 0,
        "embedding_model": EMBED_MODEL_NAME,
        "default_model": DEFAULT_MODEL_CHOICE,
        "model_choices": [
            {"value": value, "label": MODEL_DISPLAY_NAMES.get(value, value)}
            for value in MODEL_CHOICES
        ],
        "models": get_system_models_info(),
        "version": __version__,
    }


if __name__ == "__main__":
    import uvicorn
    port = next((p for p in [17995, 17996, 17997, 17998, 17999] if is_port_available(p)), 17995)
    logger.info(f"启动API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
