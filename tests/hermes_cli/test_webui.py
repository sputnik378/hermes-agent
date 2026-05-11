from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture
def webui_mod(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import importlib
    import hermes_cli.webui as mod

    importlib.reload(mod)
    return mod


def test_module_exports_command(webui_mod):
    assert callable(webui_mod.webui_command)


def test_ensure_env_defaults_preserves_secrets_and_merges_cors(webui_mod, monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "keep-api-key")
    monkeypatch.setenv("HERMES_WEBUI_PASSWORD", "keep-password")
    monkeypatch.setenv("API_SERVER_CORS_ORIGINS", "https://example.com,http://localhost:8787")

    info = webui_mod._ensure_env_defaults()

    env_path = Path(webui_mod.get_hermes_home()) / ".env"
    content = env_path.read_text()
    assert "API_SERVER_KEY=keep-api-key" in content
    assert "HERMES_WEBUI_PASSWORD=keep-password" in content
    assert "API_SERVER_ENABLED=true" in content
    assert "API_SERVER_HOST=127.0.0.1" in content
    assert "API_SERVER_PORT=8642" in content
    assert "HERMES_WEBUI_HOST=127.0.0.1" in content
    assert "HERMES_WEBUI_PORT=8787" in content
    assert "API_SERVER_CORS_ORIGINS=https://example.com,http://localhost:8787,http://127.0.0.1:8787" in content
    assert info["webui_url"] == "http://127.0.0.1:8787"


def test_stop_webui_only_stops_gateway_when_marker_exists(webui_mod, monkeypatch, capsys):
    called = {"gateway": 0, "pid": []}

    monkeypatch.setattr(webui_mod, "_ensure_env_defaults", lambda: {
        "webui_host": "127.0.0.1",
        "webui_port": "8787",
        "api_host": "127.0.0.1",
        "api_port": "8642",
        "webui_url": "http://127.0.0.1:8787",
        "webui_health": "http://127.0.0.1:8787/health",
        "api_health": "http://127.0.0.1:8642/health",
    })
    monkeypatch.setattr(webui_mod, "_read_pid", lambda: 12345)
    monkeypatch.setattr(webui_mod, "_find_listener_pid", lambda port: None)
    monkeypatch.setattr(webui_mod, "_terminate_pid", lambda pid: called["pid"].append(pid) or True)
    monkeypatch.setattr(webui_mod, "stop_profile_gateway", lambda: called.__setitem__("gateway", called["gateway"] + 1) or True)

    webui_mod._ensure_dirs()
    webui_mod.stop_webui()
    assert called["pid"] == [12345]
    assert called["gateway"] == 0

    webui_mod._gateway_marker_file().write_text("started-by-webui\n")
    webui_mod.stop_webui()
    assert called["pid"] == [12345, 12345]
    assert called["gateway"] == 1

    out = capsys.readouterr().out
    assert "Stopped Hermes WebUI PID 12345" in out
    assert "Stopped Hermes gateway/API server started by WebUI" in out


def test_webui_command_dispatches_start(webui_mod, monkeypatch):
    called = {}
    monkeypatch.setattr(webui_mod, "start_webui", lambda update, open_browser: called.update({"update": update, "open_browser": open_browser}))
    args = argparse.Namespace(webui_command="start", update=True, open=True)
    webui_mod.webui_command(args)
    assert called == {"update": True, "open_browser": True}
