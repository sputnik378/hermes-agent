#!/usr/bin/env bash
# =============================================================================
# matt-update.sh — Hermes fork update SOP
#
# Merges upstream NousResearch/hermes-agent into sputnik378/hermes-agent while
# preserving Matt's customizations, then pushes the result to the fork.
#
# Usage:
#   ./scripts/matt-update.sh             # interactive (confirm before push)
#   ./scripts/matt-update.sh --yes       # non-interactive (CI / 1-click)
#   ./scripts/matt-update.sh --dry-run   # fetch + report, no merge or push
#
# Custom files watched for conflicts:
#   hermes_cli/banner.py       (fork-aware update check + PyPI check)
#   hermes_cli/webui.py        (nesquena/hermes-webui sidecar manager)
#   hermes_cli/main.py         (hermes webui subcommand)
#   gateway/platforms/api_server.py  (skills/profiles/sessions/crons/transcribe endpoints)
#   hermes_cli/models.py       (Codex/ChatGPT backend probing)
#   hermes_cli/model_switch.py (OAuth external credential error handling)
#   tests/hermes_cli/test_update_check.py
#   tests/hermes_cli/test_webui.py
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
ASSUME_YES=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --yes|-y)    ASSUME_YES=1 ;;
    --dry-run)   DRY_RUN=1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED=$'\033[0;31m'; YELLOW=$'\033[1;33m'; GREEN=$'\033[0;32m'
CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; RESET=$'\033[0m'

info()    { echo "${CYAN}→${RESET} $*"; }
success() { echo "${GREEN}✓${RESET} $*"; }
warn()    { echo "${YELLOW}⚠${RESET} $*"; }
die()     { echo "${RED}✗${RESET} $*" >&2; exit 1; }
header()  { echo; echo "${BOLD}$*${RESET}"; echo "${BOLD}$(printf '─%.0s' $(seq 1 ${#1}))${RESET}"; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
header "Hermes Fork Update"

[[ -d ".git" ]] || die "Not a git repo: $REPO_DIR"

ORIGIN_URL="$(git config --get remote.origin.url 2>/dev/null || true)"
UPSTREAM_URL="$(git config --get remote.upstream.url 2>/dev/null || true)"

[[ -n "$ORIGIN_URL" ]]   || die "No 'origin' remote configured."
[[ -n "$UPSTREAM_URL" ]] || die "No 'upstream' remote configured. Run: git remote add upstream https://github.com/NousResearch/hermes-agent.git"

info "Fork:     $ORIGIN_URL"
info "Upstream: $UPSTREAM_URL"
echo

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURRENT_BRANCH" == "main" ]] || {
  warn "Currently on branch '$CURRENT_BRANCH', switching to main..."
  git checkout main
}

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
  warn "Uncommitted changes detected — stashing..."
  STASH_REF="$(git stash create "matt-update pre-stash $(date +%Y%m%dT%H%M%S)")"
  git stash store -m "matt-update pre-stash" "$STASH_REF"
  STASHED=1
else
  STASHED=0
fi

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
header "Pre-update snapshot"
SNAPSHOT_LABEL="pre-update-$(date +%Y%m%dT%H%M%SZ)"
if command -v hermes &>/dev/null && [[ $DRY_RUN -eq 0 ]]; then
  if hermes snapshot create "$SNAPSHOT_LABEL" 2>/dev/null; then
    success "Snapshot: $SNAPSHOT_LABEL"
  else
    warn "Snapshot skipped (hermes not running or snapshot command unavailable)"
  fi
else
  # Lightweight fallback: tag the current HEAD
  git tag -f "matt-pre-update-$(date +%Y%m%dT%H%M%SZ)" HEAD 2>/dev/null && \
    success "Git tag backup created at HEAD"
fi

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
header "Fetching remotes"
info "Fetching upstream..."
git fetch upstream --quiet && success "upstream fetched"
info "Fetching origin (fork)..."
git fetch origin --quiet   && success "origin fetched"

LOCAL_REV="$(git rev-parse --short HEAD)"
UPSTREAM_REV="$(git rev-parse --short upstream/main)"
BEHIND="$(git rev-list --count HEAD..upstream/main)"
AHEAD="$(git rev-list --count upstream/main..HEAD)"

echo
info "Local HEAD:     $LOCAL_REV"
info "Upstream/main:  $UPSTREAM_REV"
info "Ahead (custom): $AHEAD commit(s)"
info "Behind:         $BEHIND commit(s)"

if [[ $BEHIND -eq 0 ]]; then
  success "Already up to date with upstream."
  [[ $STASHED -eq 1 ]] && git stash pop
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  info "Dry-run mode — showing new upstream commits only, no merge performed."
  echo
  git log --oneline HEAD..upstream/main | head -20
  TOTAL="$(git rev-list --count HEAD..upstream/main)"
  [[ $TOTAL -gt 20 ]] && echo "  ... and $((TOTAL - 20)) more"
  exit 0
fi

# ---------------------------------------------------------------------------
# Show what's coming
# ---------------------------------------------------------------------------
header "New upstream commits ($BEHIND)"
git log --oneline HEAD..upstream/main | head -20
TOTAL="$(git rev-list --count HEAD..upstream/main)"
[[ $TOTAL -gt 20 ]] && echo "  ... and $((TOTAL - 20)) more"
echo

# Warn if any of the custom files are in the incoming diff
CUSTOM_FILES=(
  "hermes_cli/banner.py"
  "hermes_cli/webui.py"
  "hermes_cli/main.py"
  "gateway/platforms/api_server.py"
  "hermes_cli/models.py"
  "hermes_cli/model_switch.py"
  "tests/hermes_cli/test_update_check.py"
  "tests/hermes_cli/test_webui.py"
)
TOUCHED=()
for f in "${CUSTOM_FILES[@]}"; do
  if git diff --name-only HEAD..upstream/main | grep -qF "$f"; then
    TOUCHED+=("$f")
  fi
done

if [[ ${#TOUCHED[@]} -gt 0 ]]; then
  warn "The following custom files were changed upstream — conflicts possible:"
  for f in "${TOUCHED[@]}"; do echo "    $f"; done
  echo
fi

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------
if [[ $ASSUME_YES -eq 0 ]]; then
  read -r -p "${YELLOW}Merge upstream/main into local main?${RESET} [y/N] " CONFIRM
  [[ "$CONFIRM" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
fi

# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
header "Merging upstream/main"
MERGE_MSG="Merge upstream $(git describe --tags upstream/main 2>/dev/null || echo "$(date +%Y-%m-%d)") + Matt customizations"

if git merge upstream/main --no-edit -m "$MERGE_MSG" 2>&1; then
  success "Merged cleanly."
else
  echo
  warn "Merge conflicts — resolve these files, then run:"
  echo
  git diff --name-only --diff-filter=U | while read -r f; do
    echo "    ${RED}CONFLICT${RESET}: $f"
  done
  echo
  echo "  git add <resolved-files>"
  echo "  git commit --no-edit"
  echo "  $0 --yes          # re-run to push"
  echo
  [[ $STASHED -eq 1 ]] && warn "Your pre-merge stash is still saved — restore with: git stash pop"
  exit 1
fi

# ---------------------------------------------------------------------------
# Syntax check on customized files
# ---------------------------------------------------------------------------
header "Syntax check"
FAIL=0
for f in hermes_cli/banner.py hermes_cli/webui.py hermes_cli/main.py; do
  if [[ -f "$f" ]]; then
    if python3 -c "import ast, pathlib; ast.parse(pathlib.Path('$f').read_text())" 2>/dev/null; then
      success "$f"
    else
      warn "$f has a syntax error — check before pushing"
      FAIL=1
    fi
  fi
done
[[ $FAIL -eq 1 ]] && die "Syntax errors found. Fix them, then push manually."

# ---------------------------------------------------------------------------
# Push to fork
# ---------------------------------------------------------------------------
header "Pushing to fork"
if [[ $ASSUME_YES -eq 0 ]]; then
  NEW_HEAD="$(git rev-parse --short HEAD)"
  info "About to push $ORIGIN_URL main ($OLD_HEAD → $NEW_HEAD)"
  read -r -p "${YELLOW}Push to fork?${RESET} [y/N] " CONFIRM
  [[ "$CONFIRM" =~ ^[Yy]$ ]] || { info "Skipping push. Run: git push origin main"; exit 0; }
fi

git push origin main
success "Fork updated: $ORIGIN_URL"

# ---------------------------------------------------------------------------
# Post-merge cleanup
# ---------------------------------------------------------------------------
header "Post-merge cleanup"

# Clear update check cache so the banner refreshes immediately
UPDATE_CACHE="${HERMES_HOME:-$HOME/.hermes}/.update_check"
[[ -f "$UPDATE_CACHE" ]] && rm -f "$UPDATE_CACHE" && success "Update check cache cleared"

# Clear BWS disk cache if it exists (secrets may have rotated)
BWS_CACHE="${HERMES_HOME:-$HOME/.hermes}/cache/bws_cache.json"
[[ -f "$BWS_CACHE" ]] && rm -f "$BWS_CACHE" && success "BWS disk cache cleared"

# Pop any pre-merge stash
if [[ $STASHED -eq 1 ]]; then
  info "Restoring pre-merge stash..."
  git stash pop && success "Stash restored" || warn "Stash pop had conflicts — restore manually: git stash pop"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Done"
NEW_REV="$(git rev-parse --short HEAD)"
CUSTOM_AHEAD="$(git rev-list --count upstream/main..HEAD)"
echo
success "Local main:  $NEW_REV"
success "Fork:        in sync with local"
success "Custom commits on top: $CUSTOM_AHEAD"
echo
info "New upstream commits merged: $BEHIND"
echo
info "To verify: hermes --version"
info "To run:    hermes"
echo
