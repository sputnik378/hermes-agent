from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from hermes_cli.config import get_env_value, get_hermes_home, save_env_value
from hermes_cli.gateway import stop_profile_gateway

WEBUI_REPO_URL = "https://github.com/nesquena/hermes-webui.git"
DEFAULT_WEBUI_HOST = "127.0.0.1"
DEFAULT_WEBUI_PORT = 8787
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8642


def _webui_root() -> Path:
    return get_hermes_home() / "webui"


def _repo_dir() -> Path:
    return _webui_root() / "hermes-webui"


def _state_dir() -> Path:
    return _webui_root() / "state"


def _pid_file() -> Path:
    return _webui_root() / "webui.pid"


def _log_file() -> Path:
    return _webui_root() / "webui.log"


def _gateway_log_file() -> Path:
    return _webui_root() / "gateway-api.log"


def _gateway_marker_file() -> Path:
    return _webui_root() / "gateway.started-by-webui"


def _service_dir() -> Path:
    return _webui_root() / "service"


def _service_start_script() -> Path:
    return _service_dir() / "start-webui.sh"


def _service_stop_script() -> Path:
    return _service_dir() / "stop-webui.sh"


def _launch_agent_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "ai.hermes.webui.plist"


def _print(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dirs() -> None:
    for path in (_webui_root(), _state_dir(), _service_dir()):
        path.mkdir(parents=True, exist_ok=True)


def _discover_agent_dir() -> Path:
    env_override = os.getenv("HERMES_WEBUI_AGENT_DIR", "").strip()
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.append(get_hermes_home() / "hermes-agent")
    candidates.append(Path(__file__).resolve().parent.parent)
    for candidate in candidates:
        if (candidate / "run_agent.py").exists():
            return candidate.resolve()
    return Path(__file__).resolve().parent.parent


def _discover_python(repo_dir: Path | None = None) -> str:
    env_python = os.getenv("HERMES_WEBUI_PYTHON", "").strip()
    if env_python:
        return env_python

    agent_dir = _discover_agent_dir()
    for candidate in [
        agent_dir / "venv" / "bin" / "python",
        agent_dir / ".venv" / "bin" / "python",
    ]:
        if candidate.exists():
            return str(candidate)

    if repo_dir:
        for candidate in [repo_dir / ".venv" / "bin" / "python"]:
            if candidate.exists():
                return str(candidate)

    found = shutil.which("python3") or shutil.which("python")
    return found or sys.executable


def _http_get_json(url: str, timeout: float = 2.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # nosec B310
        return resp.read().decode("utf-8", errors="replace")


def _healthcheck(url: str, timeout: float = 2.0) -> bool:
    try:
        body = _http_get_json(url, timeout=timeout)
    except Exception:
        return False
    return '"status"' in body and 'ok' in body


def _wait_for_health(url: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _healthcheck(url):
            return True
        time.sleep(0.4)
    return False


def _csv_merge(existing: str | None, required: list[str]) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for raw in (existing or "").split(","):
        item = raw.strip()
        if item and item not in seen:
            seen.add(item)
            values.append(item)
    for item in required:
        if item and item not in seen:
            seen.add(item)
            values.append(item)
    return ",".join(values)


def _ensure_env_defaults(
    *,
    webui_host: str = DEFAULT_WEBUI_HOST,
    webui_port: int = DEFAULT_WEBUI_PORT,
    api_host: str = DEFAULT_API_HOST,
    api_port: int = DEFAULT_API_PORT,
) -> dict[str, str]:
    required_origins = [
        f"http://localhost:{webui_port}",
        f"http://127.0.0.1:{webui_port}",
    ]
    existing_api_key = get_env_value("API_SERVER_KEY")
    existing_password = get_env_value("HERMES_WEBUI_PASSWORD")
    merged_cors = _csv_merge(get_env_value("API_SERVER_CORS_ORIGINS"), required_origins)

    save_env_value("API_SERVER_ENABLED", "true")
    save_env_value("API_SERVER_HOST", api_host)
    save_env_value("API_SERVER_PORT", str(api_port))
    save_env_value("HERMES_WEBUI_HOST", webui_host)
    save_env_value("HERMES_WEBUI_PORT", str(webui_port))
    save_env_value("API_SERVER_CORS_ORIGINS", merged_cors)
    if existing_api_key:
        save_env_value("API_SERVER_KEY", existing_api_key)
    else:
        save_env_value("API_SERVER_KEY", secrets.token_urlsafe(32))
    if existing_password:
        save_env_value("HERMES_WEBUI_PASSWORD", existing_password)
    else:
        save_env_value("HERMES_WEBUI_PASSWORD", secrets.token_urlsafe(24))

    return {
        "webui_host": webui_host,
        "webui_port": str(webui_port),
        "api_host": api_host,
        "api_port": str(api_port),
        "webui_url": f"http://{webui_host}:{webui_port}",
        "webui_health": f"http://{webui_host}:{webui_port}/health",
        "api_health": f"http://{api_host}:{api_port}/health",
    }


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True, timeout=120)


def install_webui(*, update: bool = False) -> Path:
    _ensure_dirs()
    repo_dir = _repo_dir()
    if repo_dir.exists():
        if update:
            result = _run(["git", "-C", str(repo_dir), "pull", "--ff-only"])
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git pull failed")
            _print(f"Updated Hermes WebUI repo at {repo_dir}")
        else:
            _print(f"Hermes WebUI already installed at {repo_dir}")
        return repo_dir

    result = _run(["git", "clone", WEBUI_REPO_URL, str(repo_dir)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "git clone failed")
    _print(f"Installed Hermes WebUI repo at {repo_dir}")
    return repo_dir


def _read_pid() -> int | None:
    pid_path = _pid_file()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    _pid_file().write_text(f"{pid}\n")


def _pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _find_listener_pid(port: int) -> int | None:
    lsof = shutil.which("lsof")
    if not lsof:
        return None
    result = subprocess.run(
        [lsof, "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        text=True,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _terminate_pid(pid: int, *, timeout: float = 8.0) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_running(pid):
            return True
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    time.sleep(0.25)
    return not _pid_running(pid)


def _spawn_gateway(agent_dir: Path, env: dict[str, str]) -> subprocess.Popen:
    _ensure_dirs()
    log_handle = open(_gateway_log_file(), "ab")
    return subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main", "gateway", "run"],
        cwd=str(agent_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _build_webui_env(env_info: dict[str, str], repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_WEBUI_HOST"] = env_info["webui_host"]
    env["HERMES_WEBUI_PORT"] = env_info["webui_port"]
    env["HERMES_WEBUI_STATE_DIR"] = str(_state_dir())
    env["HERMES_WEBUI_AGENT_DIR"] = str(_discover_agent_dir())
    env.setdefault("HERMES_HOME", str(get_hermes_home()))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def start_webui(*, update: bool = False, open_browser: bool = False) -> None:
    _ensure_dirs()
    repo_dir = install_webui(update=update)
    env_info = _ensure_env_defaults()
    env = _build_webui_env(env_info, repo_dir)
    agent_dir = _discover_agent_dir()

    if not _healthcheck(env_info["api_health"]):
        _print("API server is not healthy; starting Hermes gateway/API server...")
        _spawn_gateway(agent_dir, env)
        _gateway_marker_file().write_text("started-by-webui\n")
        if not _wait_for_health(env_info["api_health"], timeout=30.0):
            raise RuntimeError(f"API server did not become healthy at {env_info['api_health']}")
    else:
        if _gateway_marker_file().exists():
            _gateway_marker_file().unlink()

    if _healthcheck(env_info["webui_health"]):
        pid = _find_listener_pid(int(env_info["webui_port"])) or _read_pid()
        if pid:
            _write_pid(pid)
        _print(f"Hermes WebUI already running at {env_info['webui_url']}")
        if open_browser:
            import webbrowser
            webbrowser.open(env_info["webui_url"])
        return

    python_exe = _discover_python(repo_dir)
    log_handle = open(_log_file(), "ab")
    proc = subprocess.Popen(
        [python_exe, str(repo_dir / "server.py")],
        cwd=str(repo_dir),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _write_pid(proc.pid)
    if not _wait_for_health(env_info["webui_health"], timeout=30.0):
        raise RuntimeError(f"WebUI did not become healthy at {env_info['webui_health']}")

    listener_pid = _find_listener_pid(int(env_info["webui_port"]))
    if listener_pid:
        _write_pid(listener_pid)
    _print(f"Hermes WebUI ready at {env_info['webui_url']}")
    _print(f"Log: {_log_file()}")
    if open_browser:
        import webbrowser
        webbrowser.open(env_info["webui_url"])


def stop_webui() -> None:
    env_info = _ensure_env_defaults()
    pid = _find_listener_pid(int(env_info["webui_port"])) or _read_pid()
    stopped = False
    if pid:
        stopped = _terminate_pid(pid)
        if stopped:
            _print(f"Stopped Hermes WebUI PID {pid}")
        else:
            raise RuntimeError(f"Failed to stop Hermes WebUI PID {pid}")
    else:
        _print("Hermes WebUI is not running")

    try:
        _pid_file().unlink()
    except FileNotFoundError:
        pass

    if _gateway_marker_file().exists():
        try:
            if stop_profile_gateway():
                _print("Stopped Hermes gateway/API server started by WebUI")
        finally:
            try:
                _gateway_marker_file().unlink()
            except FileNotFoundError:
                pass


def _status_line(name: str, ok: bool, detail: str) -> str:
    state = "running" if ok else "stopped"
    return f"{name}: {state} - {detail}"


def status_webui() -> None:
    env_info = _ensure_env_defaults()
    webui_pid = _find_listener_pid(int(env_info["webui_port"])) or _read_pid()
    api_pid = _find_listener_pid(int(env_info["api_port"]))
    webui_ok = _healthcheck(env_info["webui_health"])
    api_ok = _healthcheck(env_info["api_health"])

    _print(_status_line("WebUI", webui_ok, f"url={env_info['webui_url']} pid={webui_pid or '-'} log={_log_file()}"))
    _print(_status_line("API", api_ok, f"url=http://{env_info['api_host']}:{env_info['api_port']} pid={api_pid or '-'} log={_gateway_log_file()}"))
    _print(f"Repo: {_repo_dir()}")
    _print(f"State: {_state_dir()}")
    _print(f"Managed launch agent: {_launch_agent_path()}")


def _write_helper_scripts() -> tuple[Path, Path]:
    _ensure_dirs()
    start_script = _service_start_script()
    stop_script = _service_stop_script()
    repo_dir = _repo_dir()
    start_script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"cd {repo_dir}\n"
        f"exec {shutil.which('hermes') or 'hermes'} webui start\n"
    )
    stop_script.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f"exec {shutil.which('hermes') or 'hermes'} webui stop\n"
    )
    start_script.chmod(0o755)
    stop_script.chmod(0o755)
    return start_script, stop_script


def setup_launch_agent() -> Path:
    start_script, _ = _write_helper_scripts()
    plist_path = _launch_agent_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = _webui_root() / "launchd.log"
    label = "ai.hermes.webui"
    plist_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
        "<plist version=\"1.0\">\n"
        "<dict>\n"
        f"  <key>Label</key><string>{label}</string>\n"
        "  <key>RunAtLoad</key><true/>\n"
        "  <key>KeepAlive</key><true/>\n"
        "  <key>ProgramArguments</key>\n"
        "  <array>\n"
        f"    <string>{start_script}</string>\n"
        "  </array>\n"
        f"  <key>WorkingDirectory</key><string>{_repo_dir()}</string>\n"
        f"  <key>StandardOutPath</key><string>{log_path}</string>\n"
        f"  <key>StandardErrorPath</key><string>{log_path}</string>\n"
        "</dict>\n"
        "</plist>\n"
    )
    return plist_path


def webui_command(args) -> None:
    command = getattr(args, "webui_command", None)
    if command == "install":
        install_webui(update=bool(getattr(args, "update", False)))
        return
    if command == "start":
        start_webui(update=bool(getattr(args, "update", False)), open_browser=bool(getattr(args, "open", False)))
        return
    if command == "stop":
        stop_webui()
        return
    if command == "status":
        status_webui()
        return
    raise SystemExit("Usage: hermes webui {install,start,stop,status}")
