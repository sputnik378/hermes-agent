import subprocess


def test_fork_banner_counts_carried_commits_from_upstream(tmp_path):
    from hermes_cli.banner import get_git_banner_state

    def git(*args):
        return subprocess.check_output(
            ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
             "-c", "commit.gpgsign=false", *args], cwd=tmp_path, text=True,
        ).strip()

    git("init", "-b", "main")
    git("commit", "--allow-empty", "-m", "upstream")
    base = git("rev-parse", "HEAD")
    git("remote", "add", "upstream", "https://example.invalid/upstream.git")
    git("update-ref", "refs/remotes/upstream/main", base)
    git("commit", "--allow-empty", "-m", "carried")
    local = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", local)

    assert get_git_banner_state(tmp_path) == {
        "upstream": base[:8], "local": local[:8], "ahead": 1,
    }
