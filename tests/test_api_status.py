import asyncio

import api_router
from api_router import check_status
from version import __version__


def test_status_reports_current_version_without_credentials(monkeypatch):
    # 隔离本机 .env 中的真实凭证，确保独立验证「未配置凭证」场景
    monkeypatch.setattr(api_router, "SILICONFLOW_API_KEY", None)
    monkeypatch.setattr(api_router, "MAGICK_API_KEY", None)
    monkeypatch.setattr(api_router, "check_serpapi_key", lambda: False)

    status = asyncio.run(check_status())

    assert status["status"] == "healthy"
    assert status["version"] == __version__
    assert status["siliconflow_configured"] is False
    assert status["magick_configured"] is False
