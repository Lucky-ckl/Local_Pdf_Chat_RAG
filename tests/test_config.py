import config

import os
import subprocess
import sys
from pathlib import Path


def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True


def test_default_model_selection_order():
    assert config.choose_default_model("sk-silicon", "sk-magick", True) == "siliconflow"
    assert config.choose_default_model(None, "sk-magick", True) == "magick"
    assert config.choose_default_model(None, None, True) == "ollama"
    assert config.choose_default_model(None, None, False) == "siliconflow"


def test_env_int_helper_parses_valid_and_falls_back(monkeypatch):
    monkeypatch.setenv("_TEST_INT", "123")
    assert config._env_int("_TEST_INT", 7) == 123
    monkeypatch.setenv("_TEST_INT", "  42 ")
    assert config._env_int("_TEST_INT", 7) == 42
    monkeypatch.setenv("_TEST_INT", "abc")
    assert config._env_int("_TEST_INT", 7) == 7
    monkeypatch.delenv("_TEST_INT", raising=False)
    assert config._env_int("_TEST_INT", 7) == 7


def test_env_float_helper_parses_valid_and_falls_back(monkeypatch):
    monkeypatch.setenv("_TEST_FLOAT", "0.25")
    assert config._env_float("_TEST_FLOAT", 1.0) == 0.25
    monkeypatch.setenv("_TEST_FLOAT", "oops")
    assert config._env_float("_TEST_FLOAT", 1.0) == 1.0
    monkeypatch.delenv("_TEST_FLOAT", raising=False)
    assert config._env_float("_TEST_FLOAT", 1.0) == 1.0


def test_env_bool_helper_parses_common_values(monkeypatch):
    for truthy in ["1", "true", "TRUE", "Yes", "on", "  on  "]:
        monkeypatch.setenv("_TEST_BOOL", truthy)
        assert config._env_bool("_TEST_BOOL", False) is True
    for falsy in ["0", "false", "no", "off"]:
        monkeypatch.setenv("_TEST_BOOL", falsy)
        assert config._env_bool("_TEST_BOOL", True) is False
    monkeypatch.delenv("_TEST_BOOL", raising=False)
    assert config._env_bool("_TEST_BOOL", True) is True


def test_new_ocr_and_session_config_attributes_exist():
    for name in [
        "OCR_ENABLED",
        "OCR_ENGINE",
        "OCR_RENDER_DPI",
        "SCAN_TEXT_THRESHOLD",
        "OCR_MAX_PAGES",
        "INDEX_DIR",
        "SESSION_MAX_TURNS",
        "SSE_HEARTBEAT_SEC",
    ]:
        assert hasattr(config, name), name


def test_hyperparameters_are_env_overridable():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.update({"CHUNK_SIZE": "123", "HYBRID_ALPHA": "0.25", "OCR_ENABLED": "false"})

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import config; print(config.CHUNK_SIZE, config.HYBRID_ALPHA, config.OCR_ENABLED)",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "123 0.25 False"


def test_rerank_model_name_defaults_and_is_env_overridable():
    repo_root = Path(__file__).resolve().parents[1]
    probe = "import config; print(config.RERANK_MODEL_NAME)"

    env = dict(os.environ)
    env.pop("RERANK_MODEL_NAME", None)
    default = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    assert default.returncode == 0, default.stderr
    assert default.stdout.strip() == "BAAI/bge-reranker-base"

    env["RERANK_MODEL_NAME"] = "BAAI/bge-reranker-large"
    overridden = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
    )
    assert overridden.returncode == 0, overridden.stderr
    assert overridden.stdout.strip() == "BAAI/bge-reranker-large"


def test_example_env_documents_rerank_model_name():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "example.env").read_text(encoding="utf-8")
    assert "RERANK_MODEL_NAME=BAAI/bge-reranker-base" in text
