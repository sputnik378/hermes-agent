from types import SimpleNamespace
import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("check,flag", [(False, "--yes"), (True, "--dry-run")])
def test_local_fork_update_helper_uses_extracted_git_helpers(tmp_path, monkeypatch, check, flag):
    """The fork updater must work after git helpers moved out of main.py."""
    from hermes_cli import main

    helper = tmp_path / "scripts" / "matt-update.sh"
    helper.parent.mkdir()
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    helper.chmod(0o755)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    for remote, owner in (("origin", "sputnik378"), ("upstream", "NousResearch")):
        subprocess.run(
            ["git", "remote", "add", remote, f"https://github.com/{owner}/hermes-agent.git"],
            cwd=tmp_path, check=True,
        )
    original_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd[0] == "git":
            return original_run(cmd, **kwargs)
        if cmd == [str(helper), flag]:
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        handled = main._run_local_fork_update_helper_if_available(
            SimpleNamespace(branch=None, check=check, yes=True, gateway=False)
        )

    assert handled is True
