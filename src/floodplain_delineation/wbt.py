"""Thin, strict wrapper around the WhiteboxTools executable.

Why not ``whitebox.WhiteboxTools.run_tool``? It returns 0 even when the tool panics.
This runner calls the binary directly so that:

* a non-zero exit status is always an error;
* output text beginning with "error"/"panicked" is treated as an error;
* every expected output file must exist and be non-empty;
* every flag passed is verified against the installed tool's own help text
  (WhiteboxTools silently ignores unknown flags).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .exceptions import WhiteboxError
from .logging_utils import get_logger

log = get_logger("wbt")


def locate_executable(explicit: Optional[Path] = None) -> Path:
    """Find the WhiteboxTools binary: config path > $WBT_EXE > `whitebox` package > PATH."""
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env = os.environ.get("WBT_EXE")
    if env:
        candidates.append(Path(env))
    if explicit is None and not env:
        try:
            import whitebox

            wbt = whitebox.WhiteboxTools()
            wbt.set_max_procs(18) 
            candidates.append(Path(wbt.exe_path) / wbt.exe_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            log.debug("whitebox package lookup failed: %s", exc)
        found = shutil.which("whitebox_tools")
        if found:
            candidates.append(Path(found))
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c.resolve()
    tried = ", ".join(map(str, candidates)) or "none"
    raise WhiteboxError(
        "WhiteboxTools executable not found (looked at: " + tried + "). Install it with "
        "`pip install whitebox`, put `whitebox_tools` on PATH, set the WBT_EXE environment "
        "variable, or set execution.whitebox_executable in the configuration."
    )


class WhiteboxRunner:
    def __init__(self, executable: Optional[Path] = None, timeout: Optional[float] = None) -> None:
        self.executable = locate_executable(executable)
        self.timeout = timeout
        self._help_cache: Dict[str, str] = {}
        self.calls: List[Dict[str, object]] = []  # recorded for the processing summary

    # ------------------------------------------------------------------ #
    def _exec(self, args: Sequence[str]) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [str(self.executable), *args],
                capture_output=True, text=True, timeout=self.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise WhiteboxError(f"Could not execute WhiteboxTools ({self.executable}): {exc}") from exc

    def version(self) -> str:
        proc = self._exec(["--version"])
        if proc.returncode != 0:
            raise WhiteboxError(f"`whitebox_tools --version` failed: {proc.stderr.strip()[:300]}")
        return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"

    def tool_help(self, tool: str) -> str:
        if tool not in self._help_cache:
            proc = self._exec([f"--toolhelp={tool}"])
            text = proc.stdout
            if proc.returncode != 0 or "Flag" not in text:
                raise WhiteboxError(
                    f"Tool '{tool}' is not available in the installed WhiteboxTools "
                    f"({self.executable}). Upgrade WhiteboxTools or choose another configuration."
                )
            self._help_cache[tool] = text
        return self._help_cache[tool]

    def verify_tools(self, tools: Iterable[str]) -> None:
        for t in tools:
            self.tool_help(t)

    # ------------------------------------------------------------------ #
    def run(
        self,
        tool: str,
        flags: Sequence[str],
        *,
        work_dir: Path,
        outputs: Sequence[Path],
    ) -> str:
        """Run ``tool`` with ``flags`` (e.g. ``["--dem=/a.tif", "--fix_flats"]``).

        Raises :class:`WhiteboxError` on any failure; returns the captured text output.
        """
        help_text = self.tool_help(tool)
        for flag in flags:
            name = flag.split("=", 1)[0]
            if name not in help_text:
                raise WhiteboxError(
                    f"Flag '{name}' is not documented for tool '{tool}' in the installed "
                    "WhiteboxTools; refusing to pass it (unknown flags would be silently ignored)."
                )
        for out in outputs:
            if out.exists():
                out.unlink()
        cmd = [f"--run={tool}", f"--wd={work_dir}", *flags, "-v=false"]
        log.debug("WhiteboxTools: %s", " ".join(cmd))
        proc = self._exec(cmd)
        text = (proc.stdout or "") + (proc.stderr or "")
        self.calls.append({"tool": tool, "flags": list(flags), "returncode": proc.returncode})
        if proc.returncode != 0:
            raise WhiteboxError(
                f"WhiteboxTools '{tool}' failed with exit status {proc.returncode}. "
                f"Output: {text.strip()[:600] or '(none)'}"
            )
        for line in text.splitlines():
            if line.strip().lower().startswith(("error", "thread 'main' panicked")):
                raise WhiteboxError(f"WhiteboxTools '{tool}' reported an error: {line.strip()[:400]}")
        for out in outputs:
            if not out.is_file() or out.stat().st_size == 0:
                raise WhiteboxError(
                    f"WhiteboxTools '{tool}' finished without creating the expected output {out}."
                )
        return text
