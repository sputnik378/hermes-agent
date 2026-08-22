from unittest.mock import MagicMock, patch




def test_format_banner_version_label_on_upstream_main():
    from hermes_cli import banner

    with patch.object(
        banner,
        "get_git_banner_state",
        return_value={"upstream": "b2f477a3", "local": "b2f477a3", "ahead": 0},
    ):
        value = banner.format_banner_version_label()

    assert value.endswith("· upstream b2f477a3")
    assert "local" not in value


def test_get_git_banner_state_reads_origin_and_head(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "remote"): MagicMock(returncode=0, stdout="origin\n"),
        ("git", "rev-parse", "--short=8", "origin/main"): MagicMock(returncode=0, stdout="b2f477a3\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="af8aad31\n"),
        ("git", "rev-list", "--count", "origin/main..HEAD"): MagicMock(returncode=0, stdout="3\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "b2f477a3", "local": "af8aad31", "ahead": 3}


def test_check_via_local_git_ssh_fastpath_ahead_not_behind(tmp_path):
    """SSH fast path must not report an ahead (carried) HEAD as behind.

    A carried local commit means tip SHAs differ, but the fresh upstream tip
    is an ancestor of HEAD — that is "ahead", and reporting it as behind
    nudges the user into `hermes update`, which can wipe the carried work.
    """
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40  # carried commit, differs from upstream tip
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        # merge-base --is-ancestor exits 0: upstream tip IS an ancestor of HEAD
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=0)),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 0


def test_get_git_banner_state_prefers_upstream_remote_for_forks(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    results = {
        ("git", "remote"): MagicMock(returncode=0, stdout="origin\nupstream\n"),
        ("git", "rev-parse", "--short=8", "upstream/main"): MagicMock(returncode=0, stdout="24f74eb8\n"),
        ("git", "rev-parse", "--short=8", "HEAD"): MagicMock(returncode=0, stdout="d19cd004\n"),
        ("git", "rev-list", "--count", "upstream/main..HEAD"): MagicMock(returncode=0, stdout="9\n"),
    }

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        if key not in results:
            raise AssertionError(f"unexpected command: {cmd}")
        return results[key]

    with patch("hermes_cli.banner.subprocess.run", side_effect=fake_run):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "24f74eb8", "local": "d19cd004", "ahead": 9}


def test_check_via_local_git_ssh_fastpath_genuinely_behind(tmp_path):
    """SSH fast path reports the exact count (compare API) when behind."""
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        # merge-base --is-ancestor exits 1: not an ancestor -> genuinely behind
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=3),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == 3


def test_check_via_local_git_ssh_fastpath_offline_keeps_sentinel(tmp_path):
    """Behind + compare API unreachable = honest no-count sentinel, never 1."""
    from unittest.mock import MagicMock

    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    def fake_git_stdout(args, *, cwd, timeout=5):
        if args == ["remote", "get-url", "origin"]:
            return "git@github.com:NousResearch/hermes-agent.git"
        if args == ["rev-parse", "HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    with (
        patch.object(banner, "_git_stdout", side_effect=fake_git_stdout),
        patch.object(banner, "_upstream_main_sha", return_value="a" * 40),
        patch.object(banner.subprocess, "run", return_value=MagicMock(returncode=1)),
        patch.object(banner, "_github_compare_behind", return_value=None),
    ):
        behind = banner._check_via_local_git(repo_dir)

    assert behind == banner.UPDATE_AVAILABLE_NO_COUNT


def test_get_git_banner_state_falls_back_to_build_sha_when_no_repo():
    from hermes_cli import banner

    with (
        patch.object(banner, "_git_banner_state_cache", None),
        patch.object(banner, "_resolve_repo_dir", return_value=None),
        patch("hermes_cli.build_info.get_build_sha", return_value="abcdef12"),
    ):
        state = banner.get_git_banner_state()

    assert state == {"upstream": "abcdef12", "local": "abcdef12", "ahead": 0}


def test_get_git_banner_state_returns_none_when_no_repo_and_no_build_sha():
    from hermes_cli import banner

    with (
        patch.object(banner, "_git_banner_state_cache", None),
        patch.object(banner, "_resolve_repo_dir", return_value=None),
        patch("hermes_cli.build_info.get_build_sha", return_value=None),
    ):
        state = banner.get_git_banner_state()

    assert state is None


def test_get_git_banner_state_falls_back_when_live_git_returns_nothing(tmp_path):
    from hermes_cli import banner

    repo_dir = tmp_path / "repo"
    (repo_dir / ".git").mkdir(parents=True)

    failed = MagicMock(returncode=1, stdout="")
    with (
        patch("hermes_cli.banner.subprocess.run", return_value=failed),
        patch("hermes_cli.build_info.get_build_sha", return_value="cafef00d"),
    ):
        state = banner.get_git_banner_state(repo_dir)

    assert state == {"upstream": "cafef00d", "local": "cafef00d", "ahead": 0}
