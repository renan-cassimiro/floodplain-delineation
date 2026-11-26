"""Thin, strict wrapper around the WhiteboxTools executable.

Why not ``whitebox.WhiteboxTools.run_tool``? It returns 0 even when the tool panics.
This runner calls the binary directly so that:

* a non-zero exit status is always an error;
* output text beginning with "error"/"panicked" is treated as an error;
* every expected output file must exist and be non-empty;
* every flag passed is verified against the installed tool's own help text
  (WhiteboxTools silently ignores unknown flags);
* long runs are visible: output is streamed to the log and a heartbeat (elapsed time and
  memory of the tool process) is logged periodically;
* the maximum number of processors can be set (``--max_procs``);
* raster outputs are rewritten by GDAL before another WhiteboxTools tool reads them
  (:meth:`WhiteboxRunner.run_raster`), because the GeoTIFF files written by some
  WhiteboxTools versions are not always readable by WhiteboxTools itself on very large
  grids (observed: ``Error converting TAG(34737) to ASCII`` in ``ifd.rs``).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from . import resources
from .exceptions import WhiteboxError
from .logging_utils import get_logger
from .rasters import normalize_geotiff

log = get_logger("wbt")

#: WhiteboxTools versions covered by this project's automated tests. Other versions work
#: only as far as the per-stage validations (which always run) can tell; a warning is
#: added to the processing summary. Pin the exact version with ``execution.whitebox_version``.
TESTED_WBT_VERSIONS = ("2.0.0",)

_VERSION_RE = re.compile(r"\bv(\d+(?:\.\d+)+)")


def parse_version(text: str) -> Optional[str]:
    """Extract ``'2.4.0'`` from ``'WhiteboxTools v2.4.0 (c) ...'`` (``None`` if absent)."""
    match = _VERSION_RE.search(text or "")
    return match.group(1) if match else None


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
    def __init__(
        self,
        executable: Optional[Path] = None,
        timeout: Optional[float] = None,
        *,
        max_procs: Optional[int] = None,
        heartbeat_seconds: float = 300.0,
    ) -> None:
        self.executable = locate_executable(executable)
        self.timeout = timeout
        self.max_procs = max_procs
        self.heartbeat_seconds = float(heartbeat_seconds)
        self._help_cache: Dict[str, str] = {}
        self.calls: List[Dict[str, object]] = []  # recorded for the processing summary
        #: Highest memory sampled for any tool process (``/proc``; ``None`` where unavailable or the
        #: tool finished between two samples). The kernel's ``ru_maxrss`` for children is NOT used:
        #: it is inflated by the memory of this (large) Python process at fork time.
        self.peak_rss_bytes: Optional[int] = None

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
    def _execute_streaming(self, tool: str, cmd: List[str]) -> Dict[str, object]:
        """Run the tool, stream its output to the log and log a periodic heartbeat."""
        try:
            proc = subprocess.Popen(
                [str(self.executable), *cmd],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace", bufsize=1,
            )
        except OSError as exc:
            raise WhiteboxError(f"Could not execute WhiteboxTools ({self.executable}): {exc}") from exc

        lines: List[str] = []

        def pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.append(line)
                if line.strip():
                    log.debug("WBT %s | %s", tool, line.rstrip())

        reader = threading.Thread(target=pump, name=f"wbt-{tool}-reader", daemon=True)
        reader.start()

        started = time.monotonic()
        next_beat = started + self.heartbeat_seconds
        poll = max(0.05, min(5.0, self.heartbeat_seconds))
        peak_sampled = 0
        try:
            while True:
                try:
                    proc.wait(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                rss = resources.process_rss_bytes(proc.pid)
                hwm = resources.process_peak_rss_bytes(proc.pid)
                peak_sampled = max(peak_sampled, hwm or 0, rss or 0)
                if self.timeout is not None and now - started > self.timeout:
                    proc.kill()
                    proc.wait()
                    raise WhiteboxError(
                        f"WhiteboxTools '{tool}' exceeded the timeout of {self.timeout:g} s and was killed."
                    )
                if now >= next_beat:
                    log.info(
                        "%s still running: %s elapsed, tool memory %s (peak %s).",
                        tool, resources.format_duration(now - started),
                        resources.format_bytes(rss), resources.format_bytes(hwm),
                    )
                    next_beat = now + self.heartbeat_seconds
        finally:
            if proc.poll() is None:  # interrupted (Ctrl-C, exception): never leave an orphan holding the RAM
                proc.kill()
                proc.wait()
            reader.join(timeout=10)

        elapsed = time.monotonic() - started
        return {
            "returncode": proc.returncode,
            "text": "".join(lines),
            "seconds": elapsed,
            "peak_rss_bytes_sampled": peak_sampled or None,
        }

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
        if self.max_procs is not None:  # global WhiteboxTools flag; not listed in per-tool help
            cmd.append(f"--max_procs={int(self.max_procs)}")
        log.debug("WhiteboxTools: %s", " ".join(cmd))
        result = self._execute_streaming(tool, cmd)
        text = str(result["text"])
        peak = result["peak_rss_bytes_sampled"]
        if peak:
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, int(peak))
        self.calls.append(
            {
                "tool": tool,
                "flags": list(flags),
                "returncode": result["returncode"],
                "seconds": round(float(result["seconds"]), 2),
                "peak_rss_gib_sampled": None if peak is None else round(resources.gib(peak), 2),
                "max_procs": self.max_procs,
            }
        )
        if result["returncode"] != 0:
            raise WhiteboxError(
                f"WhiteboxTools '{tool}' failed with exit status {result['returncode']}. "
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
        log.info(
            "%s finished in %s (tool memory peak %s).",
            tool, resources.format_duration(float(result["seconds"])),
            resources.format_bytes(peak) if peak else "not sampled",
        )
        return text

    def run_raster(
        self,
        tool: str,
        flags: Sequence[str],
        *,
        work_dir: Path,
        output: Path,
        normalize: bool = True,
    ) -> str:
        """Run a tool that writes one raster (``--output``) and deliver it at ``output``.

        With ``normalize`` (default) WhiteboxTools writes to a raw file inside ``work_dir``;
        that file is rewritten by GDAL into ``output`` (same grid, dtype, nodata and CRS,
        uncompressed, block by block) and removed. ``output`` is therefore never a file
        written by WhiteboxTools, so the next WhiteboxTools tool can always read it, and a
        half-written tool output can never be mistaken for a finished product.

        With ``normalize=False`` the tool writes directly to ``output`` (use only for rasters
        that no WhiteboxTools tool reads afterwards).
        """
        if output.exists():
            output.unlink()
        if not normalize:
            return self.run(tool, [*flags, f"--output={output}"], work_dir=work_dir, outputs=[output])
        raw = work_dir / f"{output.stem}.wbt_raw.tif"
        text = self.run(tool, [*flags, f"--output={raw}"], work_dir=work_dir, outputs=[raw])
        started = time.monotonic()
        try:
            normalize_geotiff(raw, output)
        finally:
            raw.unlink(missing_ok=True)
        log.info(
            "%s output rewritten with GDAL as %s in %s.",
            tool, output.name, resources.format_duration(time.monotonic() - started),
        )
        return text
