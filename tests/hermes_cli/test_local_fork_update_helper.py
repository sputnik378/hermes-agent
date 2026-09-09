from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_local_fork_update_helper_uses_extracted_git_helpers(tmp_path, monkeypatch):
    """The fork updater must work after git helpers moved out of main.py."""
    import hermes_cli.main as main

    helper = tmp_path / "scripts" / "matt-update.sh"
    helper.parent.mkdir()
    helper.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    helper.chmod(0o755)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def fake_run(cmd, **kwargs):
        if cmd == ["git", "remote", "get-url", "origin"]:
            return MagicMock(
                returncode=0,
                stdout="https://github.com/sputnik378/hermes-agent.git\n",
            )
        if cmd == ["git", "remote", "get-url", "upstream"]:
            return MagicMock(returncode=0, stdout="")
        if cmd == [str(helper), "--yes"]:
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    with patch("subprocess.run", side_effect=fake_run):
        handled = main._run_local_fork_update_helper_if_available(
            SimpleNamespace(branch=None, check=False, yes=True, gateway=False)
        )

    assert handled is True
