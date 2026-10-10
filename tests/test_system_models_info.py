"""P1.5：系统模型信息展示必须来自实际配置，而不是硬编码常量。

WBS P1.5 验收标准：修改 .env 中任一参数后，界面显示同步变化。
config 在导入时读取 .env 并绑定为模块级常量，``core.system_info`` 从这些常量读取
（P4.1：该函数从 ``rag_demo.py`` 迁出到 ``core/system_info.py``，断言保持不变），
因此这里通过 monkeypatch 改写 ``core.system_info`` 从 config 引入的名字来模拟「改了配置」，
验证展示值会随之变化（否则说明仍是硬编码）。
"""

import config
from core import system_info
from core.embeddings import EMBED_MODEL_NAME


def test_embedding_model_matches_runtime_source():
    """嵌入模型展示值应与实际驱动模型加载的常量一致。"""
    info = system_info.get_system_models_info()
    assert info["嵌入模型"] == EMBED_MODEL_NAME


def test_models_info_is_wired_to_config():
    """system_info 使用的分块/混合权重应就是 config 的值（而非另造常量）。"""
    assert system_info.CHUNK_SIZE == config.CHUNK_SIZE
    assert system_info.CHUNK_OVERLAP == config.CHUNK_OVERLAP
    assert system_info.HYBRID_ALPHA == config.HYBRID_ALPHA
    assert system_info.RERANK_METHOD == config.RERANK_METHOD
    assert system_info.RERANK_MODEL_NAME == config.RERANK_MODEL_NAME


def test_models_info_reflects_current_config_values():
    info = system_info.get_system_models_info()
    assert f"chunk_size={config.CHUNK_SIZE}" in info["分块方法"]
    assert f"overlap={config.CHUNK_OVERLAP}" in info["分块方法"]
    assert f"α={config.HYBRID_ALPHA}" in info["检索方法"]


def test_models_info_follows_config_changes(monkeypatch):
    """改配置后展示必须同步变化，且不再残留旧的硬编码值。"""
    monkeypatch.setattr(system_info, "CHUNK_SIZE", 1234)
    monkeypatch.setattr(system_info, "CHUNK_OVERLAP", 56)
    monkeypatch.setattr(system_info, "HYBRID_ALPHA", 0.42)
    info = system_info.get_system_models_info()
    assert "chunk_size=1234" in info["分块方法"]
    assert "overlap=56" in info["分块方法"]
    assert "α=0.42" in info["检索方法"]
    assert "chunk_size=400" not in info["分块方法"]
    assert "α=0.7" not in info["检索方法"]


def test_rerank_info_reflects_cross_encoder(monkeypatch):
    monkeypatch.setattr(system_info, "RERANK_METHOD", "cross_encoder")
    monkeypatch.setattr(system_info, "RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
    info = system_info.get_system_models_info()
    assert "cross_encoder" not in info["重排序模型"]
    assert "交叉编码器" in info["重排序模型"]
    assert "BAAI/bge-reranker-base" in info["重排序模型"]


def test_rerank_info_reflects_llm_method(monkeypatch):
    """RERANK_METHOD=llm 时不得仍显示「交叉编码器」。"""
    monkeypatch.setattr(system_info, "RERANK_METHOD", "llm")
    info = system_info.get_system_models_info()
    assert "交叉编码器" not in info["重排序模型"]
    assert "LLM" in info["重排序模型"]
    assert system_info.OLLAMA_MODEL_NAME in info["重排序模型"]


def test_rerank_info_reflects_disabled(monkeypatch):
    monkeypatch.setattr(system_info, "RERANK_METHOD", "none")
    info = system_info.get_system_models_info()
    assert "交叉编码器" not in info["重排序模型"]
    assert "未启用" in info["重排序模型"]


def test_rerank_info_reflects_unknown_method(monkeypatch):
    monkeypatch.setattr(system_info, "RERANK_METHOD", "some_unknown_method")
    info = system_info.get_system_models_info()
    assert "some_unknown_method" in info["重排序模型"]
    assert "交叉编码器" not in info["重排序模型"]
