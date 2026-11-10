"""Run provenance: git state, package versions, platform. Written next to every run's outputs."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TRACKED_PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy", "pandas",
    "scipy", "matplotlib", "seaborn", "pyyaml", "pytest", "bitsandbytes",
)


def package_versions() -> dict[str, str | None]:
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def collect(argv: list[str] | None = None) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": argv if argv is not None else sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_state(),
        "packages": package_versions(),
    }
