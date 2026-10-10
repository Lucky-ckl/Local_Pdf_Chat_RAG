"""文档入库编排（P4.1：从 Gradio 入口 ``rag_demo.py`` 迁出，供 REST / Web 复用）。

职责：把上传的文件按 P2.2–P2.7 的解析 / 分块 / 索引流程处理，并返回**逐文件**的处理结果
与 OCR 判定结果，供 ``POST /api/upload``（WBS v1.0 P4.2）使用。本模块不依赖任何 UI 框架。

行为沿用既有实现（不新增架构）：
- ``reset=True``（默认，供 ``POST /api/upload``）：沿用 WBS §14「全量重建」，先清空再用本次文件重建；
- ``reset=False``（供知识库目录增量同步）：**不清空**，只把新 chunk 追加到现有索引，多文档可累计检索；
- 单文件失败只记录该文件的错误并继续处理其余文件，不中断整批；
- 仅支持白名单扩展名（P4.2）。
"""

import hashlib
import logging
import os
import threading

from core.bm25_index import bm25_manager
from core.document_loader import load_document
from core.embeddings import encode_texts
from core.text_splitter import split_blocks
from core.vector_store import save_persisted_index, vector_store

# 文件类型白名单（P4.2，沿用既有 Gradio File 组件允许的类型）。
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"}


def ingest_files(paths, force_ocr=False, progress=None, reset=True):
    """处理一组本地文件路径：解析 → 分块 → 向量化 → 构建 FAISS/BM25 → 持久化。

    Args:
        paths: 本地文件路径列表（调用方负责创建与清理临时文件）。
        force_ocr: 是否强制 OCR（P2.5 / P4.2 的 ``force_ocr`` 开关）。
        progress: 可选 ``callable(value, desc=None)``，用于上报阶段进度。
        reset: ``True``（默认）先清空现有索引再重建；``False`` 增量追加，保留已有文档。

    Returns:
        逐文件结果与汇总 dict，可直接序列化为 ``POST /api/upload`` 的响应；不向上抛异常。
    """
    def _report(value, desc=None):
        if progress is not None:
            try:
                progress(value, desc)
            except Exception:
                pass

    if not paths:
        return {"status": "error", "message": "请选择要上传的文件",
                "files": [], "total_chunks": 0, "persisted": False}

    if reset:
        _report(0.05, "清理历史数据...")
        vector_store.clear()
        bm25_manager.clear()

    total_files = len(paths)
    files_result = []
    all_chunks, all_metadatas, all_ids = [], [], []

    for idx, path in enumerate(paths, 1):
        file_name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        _report((idx - 1) / total_files, f"处理文件 {idx}/{total_files}: {file_name}")
        if ext not in ALLOWED_EXTENSIONS:
            files_result.append({
                "filename": file_name, "status": "error", "chunks": 0, "ocr_used": False,
                "message": f"不支持的文件类型: {ext or '(无扩展名)'}",
            })
            continue
        try:
            # P2.2 / P2.5 / P2.7：按页加载 Block（含扫描件判定与 OCR），再按页切分，
            # 使每个 chunk 携带 source / page / chunk_index / ocr_used 元数据。
            blocks = load_document(path, force_ocr=force_ocr)
            text = "\n".join(block.text for block in blocks)
            if not text.strip():
                raise ValueError("文档内容为空或无法提取文本")

            chunks = split_blocks(blocks)
            # 文档 ID 由**绝对路径**派生（稳定且唯一）：原先的 f"doc_{时间}_{idx}" 会在同一秒内
            # 对多个文件产生相同 ID（idx 每批从 1 重新计数），导致 chunk_id 冲突、相互覆盖。
            doc_id = "doc_" + hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
            metadatas = [
                {
                    "source": chunk.source or file_name,
                    "doc_id": doc_id,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "ocr_used": chunk.ocr_used,
                }
                for chunk in chunks
            ]
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

            all_chunks.extend(chunk_texts)
            all_metadatas.extend(metadatas)
            all_ids.extend(chunk_ids)

            ocr_used = any(chunk.ocr_used for chunk in chunks)
            files_result.append({
                "filename": file_name, "status": "success", "chunks": len(chunks),
                "ocr_used": ocr_used,
                "message": f"成功处理 {len(chunks)} 个文本块"
                           + ("（已启用 OCR）" if ocr_used else "（文本型）"),
            })
        except Exception as exc:  # 单文件失败继续处理其余文件（P0 既有行为）
            logging.exception("处理文件 %s 时出错", file_name)
            files_result.append({
                "filename": file_name, "status": "error", "chunks": 0, "ocr_used": False,
                "message": f"处理失败: {exc}",
            })

    persisted = False
    if all_chunks:
        _report(0.8, "生成文本嵌入...")
        embeddings = encode_texts(all_chunks, show_progress=False)
        _report(0.9, "写入向量索引...")
        if reset:
            vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)
        else:
            # 增量：保留已有文档，仅追加本次新增 chunk（知识库目录模式，不覆盖旧文档）。
            vector_store.add_index(all_chunks, all_ids, all_metadatas, embeddings)
        _report(0.95, "构建 BM25 检索索引...")
        # BM25 需覆盖**全部** chunk（含历史文档），故用累计后的语料重建。
        bm25_manager.build_index(
            [vector_store.contents_map[cid] for cid in vector_store.id_order],
            list(vector_store.id_order),
        )
        _report(0.98, "持久化索引...")
        persisted = bool(save_persisted_index())

    _report(1.0, "完成")
    total_chunks = vector_store.total_chunks
    ok = any(item["status"] == "success" for item in files_result)
    summary = f"共处理 {total_files} 个文件，{total_chunks} 个文本块"
    if all_chunks:
        summary += "；索引已持久化" if persisted else "；索引持久化失败（本次仍可在当前进程内检索）"
    return {
        "status": "success" if ok else "error",
        "message": summary,
        "files": files_result,
        "total_chunks": total_chunks,
        "persisted": persisted,
    }


_KB_SYNC_LOCK = threading.Lock()


def _documents_dir(documents_dir=None):
    """知识库目录：显式传入优先，否则取 ``config.DOCUMENTS_DIR``（惰性导入，避免测试期加载 config）。"""
    if documents_dir:
        return documents_dir
    from config import DOCUMENTS_DIR
    return DOCUMENTS_DIR


def scan_documents(documents_dir=None):
    """列出知识库目录中受支持的文件（按文件名排序；目录不存在时返回空列表）。"""
    target = _documents_dir(documents_dir)
    if not os.path.isdir(target):
        return []
    found = []
    for name in sorted(os.listdir(target)):
        path = os.path.join(target, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in ALLOWED_EXTENSIONS:
            found.append(path)
    return found


def indexed_sources():
    """已入库的来源文件名集合（取自真实 chunk 元数据，不额外维护状态）。"""
    sources = set()
    for meta in vector_store.metadatas_map.values():
        source = (meta or {}).get("source")
        if source:
            sources.add(source)
    return sources


def sync_documents(documents_dir=None, force_ocr=False, progress=None):
    """扫描固定知识库目录，把**新增**文档增量入库（不覆盖、不清空已有文档）。

    与 ``POST /api/upload`` 的全量重建语义不同：这里只处理尚未入库的文件，新增文档只
    追加对应 chunk，已有文档保持可检索（多文档累计）。已入库判定依据真实 chunk 元数据的
    ``source``；失败的文件因未产生 chunk，下次同步会重试。

    Returns:
        dict：``{status, message, added[], skipped[], total_files, total_chunks, persisted, files}``。
    """
    with _KB_SYNC_LOCK:
        target = _documents_dir(documents_dir)
        files = scan_documents(target)
        names = [os.path.basename(path) for path in files]
        indexed = indexed_sources()
        pending = [path for path in files if os.path.basename(path) not in indexed]
        skipped = [name for name in names if name in indexed]

        if not files:
            return {"status": "success", "message": f"知识库目录中没有可用文档（{target}）",
                    "added": [], "skipped": [], "files": [], "total_files": 0,
                    "total_chunks": vector_store.total_chunks, "persisted": False}
        if not pending:
            return {"status": "success", "message": f"知识库已是最新（{len(names)} 个文档）",
                    "added": [], "skipped": skipped, "files": [], "total_files": len(names),
                    "total_chunks": vector_store.total_chunks, "persisted": False}

        result = ingest_files(pending, force_ocr=force_ocr, progress=progress, reset=False)
        result["added"] = [item["filename"] for item in result["files"]
                           if item["status"] == "success"]
        result["skipped"] = skipped
        result["total_files"] = len(names)
        result["message"] = (
            f"新增 {len(result['added'])} 个文档，跳过 {len(skipped)} 个已入库文档；"
            f"当前共 {result['total_chunks']} 个文本块"
        )
        return result


def list_documents():
    """按来源文件聚合已索引 chunk，返回文档列表（P4.2 / §11.5 ``GET /api/documents``）。

    数据来自真实 chunk 元数据（``vector_store.metadatas_map``），不虚构文档或页码。
    """
    docs = {}
    for meta in vector_store.metadatas_map.values():
        meta = meta or {}
        source = meta.get("source") or "(未知来源)"
        entry = docs.setdefault(
            source, {"source": source, "doc_id": meta.get("doc_id", ""),
                     "chunks": 0, "ocr_used": False, "pages": set()},
        )
        entry["chunks"] += 1
        if meta.get("ocr_used"):
            entry["ocr_used"] = True
        page = meta.get("page")
        if isinstance(page, int) and page > 0:
            entry["pages"].add(page)
    result = []
    for entry in docs.values():
        pages = sorted(entry.pop("pages"))
        entry["page_count"] = len(pages)
        result.append(entry)
    result.sort(key=lambda item: item["source"])
    return result
