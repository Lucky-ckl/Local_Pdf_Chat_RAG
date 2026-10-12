"""启动阶段不阻塞验收：lifespan 不得等待知识库同步完成。

回归背景：``await asyncio.to_thread(sync_documents)`` 会在大量 PDF/OCR 时阻塞启动，
导致 uvicorn 迟迟不打印 ``Application startup complete``。此模块断言启动立即返回、
同步在后台线程进行，且后台失败被记录到日志。

全部通过 monkeypatch 隔离：不加载模型、不联网、不触碰真实索引。
"""

import asyncio
import logging
import threading

import api_router


def _stub_index_state():
    return {"status": "loaded", "message": "stub", "chunks": 0}


def test_lifespan_startup_does_not_wait_for_sync(monkeypatch):
    """启动必须立即完成，即使 sync_documents 长时间不返回。"""
    started = threading.Event()
    release = threading.Event()

    def slow_sync(*args, **kwargs):
        started.set()
        # 模拟 OCR/入库长耗时：直到测试显式放行才返回
        release.wait(timeout=10)
        return {"status": "success", "message": "done"}

    monkeypatch.setattr(api_router, "sync_documents", slow_sync)
    monkeypatch.setattr(api_router, "load_persisted_index", _stub_index_state)

    async def scenario():
        cm = api_router.lifespan(api_router.app)
        # 若启动阶段仍 await sync_documents，这里会超时失败
        await asyncio.wait_for(cm.__aenter__(), timeout=3)
        try:
            # 启动已返回，但后台同步仍在进行（尚未放行）
            assert started.wait(timeout=3) is True
            assert release.is_set() is False
        finally:
            release.set()
            await cm.__aexit__(None, None, None)

    asyncio.run(scenario())


def test_background_sync_failure_is_logged(monkeypatch, caplog):
    """后台同步抛异常时必须记录错误日志，且不影响启动/关闭。"""

    def boom(*args, **kwargs):
        raise RuntimeError("sync blew up")

    monkeypatch.setattr(api_router, "sync_documents", boom)
    monkeypatch.setattr(api_router, "load_persisted_index", _stub_index_state)

    async def scenario():
        with caplog.at_level(logging.ERROR, logger="rag-api"):
            async with api_router.lifespan(api_router.app):
                for _ in range(200):
                    if any("后台同步知识库失败" in r.getMessage() for r in caplog.records):
                        break
                    await asyncio.sleep(0.01)

    asyncio.run(scenario())

    assert any(
        "后台同步知识库失败" in record.getMessage() and "sync blew up" in record.getMessage()
        for record in caplog.records
    )


def test_startup_still_logs_index_state(monkeypatch, caplog):
    """后台化改造后，启动仍记录服务启动与索引状态日志（同步功能保留）。"""
    monkeypatch.setattr(
        api_router, "sync_documents",
        lambda *a, **k: {"status": "success", "message": "ok"},
    )
    monkeypatch.setattr(api_router, "load_persisted_index", _stub_index_state)

    async def scenario():
        with caplog.at_level(logging.INFO, logger="rag-api"):
            async with api_router.lifespan(api_router.app):
                pass

    asyncio.run(scenario())

    assert any("API 服务启动" in r.getMessage() for r in caplog.records)
    assert any("索引状态" in r.getMessage() for r in caplog.records)
