"""
utils/env_info.py — Collect runtime environment metadata.

Reusable across any pipeline that needs to record reproducibility info:
  - Short git hash
  - Python version / platform
  - Installed package list (pip freeze)
"""

import os
import platform
import subprocess
import sys
from typing import Dict


def get_git_hash(root_path: str = ".") -> str:
    """Return the short git hash of HEAD, or 'no-git' on any failure."""
    try:
        result = subprocess.run(
            ["git", "-C", root_path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "no-git"


def get_pip_freeze() -> str:
    """Return the output of ``pip freeze``, or 'unavailable' on failure."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def collect(root_path: str = ".") -> Dict[str, str]:
    """
    Collect environment metadata into a dict.

    Parameters
    ----------
    root_path : str
        Repository root used to query git (defaults to CWD).

    Returns
    -------
    dict with keys: git_hash, python_version, platform, pip_freeze
    """
    return {
        "git_hash": get_git_hash(root_path),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pip_freeze": get_pip_freeze(),
    }
