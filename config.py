"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 密钥配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions"
)
MAGICK_API_KEY = os.getenv("MAGICK_API_KEY")
MAGICK_API_URL = os.getenv(
    "MAGICK_API_URL",
    "https://api.magickapi.com/v1/chat/completions"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# Ollama 格式: deepseek-r1:8b
# SiliconFlow/Magick API 格式: 使用对应平台提供的模型 ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
SILICONFLOW_MODEL_NAME = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
MAGICK_MODEL_NAME = os.getenv("MAGICK_MODEL_NAME", "gpt-4o-mini")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")
# 交叉编码器模型名（P1.2；仅 RERANK_METHOD=cross_encoder 时生效）。
# 取舍：cross_encoder 精度更高但需下载模型（默认 BAAI/bge-reranker-base，中文友好）；
#       llm 复用现行 LLM 服务、零额外下载，但更慢且分数稳定性依赖所选模型。
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")

MODEL_CHOICES = ["ollama", "siliconflow", "magick"]
MODEL_DISPLAY_NAMES = {
    "ollama": "本地 Ollama 模型",
    "siliconflow": "Cloud DeepSeek-R1 模型",
    "magick": "Magick API 模型"
}


def is_configured_api_key(api_key):
    """判断 API Key 是否为用户实际配置值。"""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))


def _env_int(name, default):
    """读取整型环境变量；缺失或非法时回退默认值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logging.warning(f"环境变量 {name}={raw!r} 不是合法整数，回退默认值 {default}")
        return default


def _env_float(name, default):
    """读取浮点环境变量；缺失或非法时回退默认值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logging.warning(f"环境变量 {name}={raw!r} 不是合法浮点数，回退默认值 {default}")
        return default


def _env_bool(name, default):
    """读取布尔环境变量（1/true/yes/on 视为真）；缺失时回退默认值。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def choose_default_model(siliconflow_key, magick_key, ollama_available=False):
    """按稳定、可测试的优先级选择默认模型后端。"""
    if is_configured_api_key(siliconflow_key):
        return "siliconflow"
    if is_configured_api_key(magick_key):
        return "magick"
    if ollama_available:
        return "ollama"
    # 保持 UI 的默认选项稳定；启动检查会给出明确的未配置提示。
    return "siliconflow"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = _env_int("CHUNK_SIZE", 400)          # 文本分块大小（字符数）
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 40)     # 相邻分块的重叠字符数
HYBRID_ALPHA = _env_float("HYBRID_ALPHA", 0.7)    # 混合检索中语义检索的权重（0-1）
RETRIEVAL_TOP_K = _env_int("RETRIEVAL_TOP_K", 10)  # 检索返回的候选文档数量
RERANK_TOP_K = _env_int("RERANK_TOP_K", 5)        # 重排序后保留的文档数量
MAX_RETRIEVAL_ITERATIONS = _env_int("MAX_RETRIEVAL_ITERATIONS", 3)  # 递归检索的最大迭代轮数

# 本次改造新增参数（P2 OCR / 索引持久化，P3 会话 / 流式）
# 默认值依据 WBS v1.0 P0.3；OCR_MAX_PAGES 未由 WBS 指定，采用保守值 50（待确认）。
OCR_ENABLED = _env_bool("OCR_ENABLED", True)        # 是否启用扫描件 OCR
OCR_ENGINE = os.getenv("OCR_ENGINE", "rapidocr")    # OCR 引擎
OCR_RENDER_DPI = _env_int("OCR_RENDER_DPI", 200)    # PDF 页渲染 DPI
SCAN_TEXT_THRESHOLD = _env_int("SCAN_TEXT_THRESHOLD", 50)  # 每页平均有效字符数阈值
OCR_MAX_PAGES = _env_int("OCR_MAX_PAGES", 50)       # 单文档 OCR 页数上限（WBS 未指定默认值）
INDEX_DIR = os.getenv("INDEX_DIR", "data/index/")   # 索引持久化目录
SESSION_MAX_TURNS = _env_int("SESSION_MAX_TURNS", 5)  # 多轮保留轮数
SSE_HEARTBEAT_SEC = _env_int("SSE_HEARTBEAT_SEC", 15)  # 流式心跳间隔（秒）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    自动检测可用的 LLM 后端，返回默认模型选择

    检测优先级：
    1. SiliconFlow API Key 已配置 → 默认使用云端 API
    2. Magick API Key 已配置 → 默认使用 Magick API
    3. 本地 Ollama 服务可用 → 默认使用本地模型
    4. 都不可用 → 返回 siliconflow 并提示用户配置
    """
    if is_configured_api_key(SILICONFLOW_API_KEY):
        logging.info("✅ 检测到 SiliconFlow API Key，默认使用云端模型")
        return choose_default_model(SILICONFLOW_API_KEY, MAGICK_API_KEY)

    if is_configured_api_key(MAGICK_API_KEY):
        logging.info("✅ 检测到 Magick API Key，默认使用 Magick API 模型")
        return choose_default_model(SILICONFLOW_API_KEY, MAGICK_API_KEY)

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logging.info("✅ 检测到本地 Ollama 服务，默认使用本地模型")
            return choose_default_model(SILICONFLOW_API_KEY, MAGICK_API_KEY, ollama_available=True)
    except Exception:
        pass

    logging.warning("⚠️ 未检测到可用 LLM 后端，请配置 SiliconFlow/Magick API Key 或启动 Ollama")
    return choose_default_model(SILICONFLOW_API_KEY, MAGICK_API_KEY)

DEFAULT_MODEL_CHOICE = detect_default_model()
