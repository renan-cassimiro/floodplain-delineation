"""Lightweight resource introspection (memory, disk, peak RSS). Linux-first, never fatal.

Every function returns ``None`` when the information is unavailable (other OS, restricted
container, ...) so that diagnostics can never break a processing run.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

try:  # pragma: no cover - not available on Windows
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None

GIB = float(2 ** 30)


def gib(n_bytes: Optional[float]) -> Optional[float]:
    return None if n_bytes is None else float(n_bytes) / GIB


def _read_proc_status_kb(pid: int, key: str) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def process_rss_bytes(pid: int) -> Optional[int]:
    """Current resident set size of ``pid`` (Linux ``/proc``)."""
    kb = _read_proc_status_kb(pid, "VmRSS")
    return None if kb is None else kb * 1024


def process_peak_rss_bytes(pid: int) -> Optional[int]:
    """Peak resident set size ("high-water mark") of ``pid`` so far (Linux ``/proc``)."""
    kb = _read_proc_status_kb(pid, "VmHWM")
    return None if kb is None else kb * 1024


def _ru_maxrss_bytes(who: int) -> Optional[int]:
    if _resource is None:
        return None
    try:
        value = _resource.getrusage(who).ru_maxrss
    except (OSError, ValueError):  # pragma: no cover
        return None
    return int(value) if sys.platform == "darwin" else int(value) * 1024


def self_peak_rss_bytes() -> Optional[int]:
    """Peak RSS of this Python process since it started."""
    return _ru_maxrss_bytes(_resource.RUSAGE_SELF) if _resource else None


def available_memory_bytes() -> Dict[str, Optional[int]]:
    """Machine ``MemAvailable`` and the container (cgroup) memory limit, when known."""
    info: Dict[str, Optional[int]] = {"machine_available": None, "cgroup_limit": None, "effective": None}
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    info["machine_available"] = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    for candidate in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            text = Path(candidate).read_text().strip()
        except OSError:
            continue
        if text.isdigit() and int(text) < 2 ** 60:  # "max" or huge values mean "no limit"
            info["cgroup_limit"] = int(text)
            break
    known = [v for v in (info["machine_available"], info["cgroup_limit"]) if v is not None]
    info["effective"] = min(known) if known else None
    return info


def free_disk_bytes(path: Path) -> Optional[int]:
    """Free space on the file system that holds ``path`` (or its nearest existing parent)."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects container CPU affinity
    except (AttributeError, OSError):  # pragma: no cover
        return os.cpu_count() or 1


def format_bytes(n_bytes: Optional[float]) -> str:
    return "unknown" if n_bytes is None else f"{n_bytes / GIB:.1f} GiB"


def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
