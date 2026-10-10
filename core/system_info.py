"""系统模型 / 参数信息（P4.1：从 Gradio 入口 ``rag_demo.py`` 迁出）。

P1.5 要求：界面展示的模型与参数必须来自当前进程**实际生效的配置**，不得硬编码。
本模块把 ``rag_demo.py::get_system_models_info`` 原样迁出，供 ``GET /api/status``（§11.5）
返回给前端；config 在导入时读取 ``.env``，故改配置后需重启进程才会同步。
"""

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    HYBRID_ALPHA,
    MAGICK_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    RERANK_METHOD,
    RERANK_MODEL_NAME,
    SILICONFLOW_MODEL_NAME,
)
from core.embeddings import EMBED_MODEL_NAME


def get_system_models_info():
    """返回系统使用的各种模型 / 参数信息（读取真实 config 与运行时来源）。"""
    if RERANK_METHOD == "cross_encoder":
        rerank_info = f"交叉编码器 ({RERANK_MODEL_NAME})"
    elif RERANK_METHOD == "llm":
        rerank_info = f"LLM 相关性评分 (Ollama: {OLLAMA_MODEL_NAME})"
    elif RERANK_METHOD == "none":
        rerank_info = "未启用（使用 RRF 融合排序）"
    else:
        rerank_info = f"未识别的方法 ({RERANK_METHOD})，运行时按服务不可用降级"
    return {
        "嵌入模型": EMBED_MODEL_NAME,
        "分块方法": f"RecursiveCharacterTextSplitter (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})",
        "检索方法": f"向量检索 + BM25混合检索 (α={HYBRID_ALPHA})",
        "重排序模型": rerank_info,
        "生成模型(Ollama)": OLLAMA_MODEL_NAME,
        "生成模型(SiliconFlow)": SILICONFLOW_MODEL_NAME,
        "生成模型(Magick API)": MAGICK_MODEL_NAME,
        "分词工具": "jieba (中文分词)",
    }
