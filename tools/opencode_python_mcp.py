#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from fastmcp import FastMCP


DEFAULT_TIMEOUT_MS = 60_000


def _timeout_ms() -> int:
    raw = os.environ.get("MLEVOLVE_OPENCODE_PYTHON_TIMEOUT_MS", "")
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_MS


mcp = FastMCP("mlevolve-python")


@mcp.tool
def python_exec(args: list[str]) -> str:
    """Run python3 with explicit argv under a hard timeout in the ml environment."""
    if not all(isinstance(item, str) for item in args):
        raise ValueError("args must be a list of strings")

    cwd = Path.cwd()
    timeout_ms = _timeout_ms()
    cmd = ["micromamba", "run", "-n", "ml", "python3", *args]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout_ms / 1000.0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
        return (
            f"command: {cmd}\n"
            f"exit_code: {proc.returncode}\n\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    except subprocess.TimeoutExpired as exc:
        return (
            f"command: {cmd}\n"
            "warning: python_exec timed out\n"
            "note: a timeout on the default full-scale path can be healthy and expected.\n"
            "note: a timeout on `--check` is worrying; the `--check` path should stay lightweight.\n\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        )


if __name__ == "__main__":
    mcp.run()
