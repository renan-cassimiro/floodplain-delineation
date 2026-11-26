"""Stage manifest: lets an interrupted or partially changed run resume instead of restarting.

Each heavy stage records a *fingerprint* (hash of everything that determines its output: the
parameters that matter for that stage, the fingerprints of the stages it depends on, the input
file signature and the WhiteboxTools version), the signature of its output files and the
diagnostics it produced. On ``execution.resume: true`` a stage is skipped only when

* a manifest entry exists with an identical fingerprint, and
* every output file of the stage exists with the recorded size.

Changing only the HAND threshold therefore re-runs just the final stage; changing the drainage
threshold re-runs drainage + HAND; changing the DEM, the window or the conditioning method
re-runs everything that depends on them.

Limits (deliberate, documented): the DEM is identified by path, size and modification time, not
by content hash (hashing a multi-GB raster would take minutes); intermediates edited by hand are
not detected beyond their size.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .logging_utils import get_logger

log = get_logger("manifest")

#: Bump when stage outputs change meaning, so older manifests are never reused.
MANIFEST_SCHEMA = 2

MANIFEST_NAME = "stage_manifest.json"


def file_signature(path: Path) -> Dict[str, Any]:
    """Path, size and mtime (ns) of a file; used to fingerprint large inputs cheaply."""
    st = Path(path).stat()
    return {"path": str(Path(path).resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(stage: str, params: Mapping[str, Any], upstream: Iterable[str] = ()) -> str:
    payload = {"schema": MANIFEST_SCHEMA, "stage": stage, "params": params, "upstream": sorted(upstream)}
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class StageManifest:
    """JSON file with one entry per completed stage (atomic writes)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        self.entries = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("schema") == MANIFEST_SCHEMA and isinstance(data.get("stages"), dict):
            self.entries = data["stages"]
        else:
            log.warning("Ignoring stage manifest %s (unknown schema).", self.path)

    def save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(
            json.dumps({"schema": MANIFEST_SCHEMA, "stages": self.entries}, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------ #
    def invalidate(self, stage: str) -> None:
        """Forget a stage *before* (re)running it, so a crash never leaves a stale 'valid' entry."""
        if self.entries.pop(stage, None) is not None:
            self.save()

    def record(
        self, stage: str, fp: str, outputs: Iterable[Path], result: Mapping[str, Any], seconds: float,
        warnings: Iterable[str] = (),
    ) -> None:
        self.entries[stage] = {
            "fingerprint": fp,
            "outputs": {str(p): Path(p).stat().st_size for p in outputs if Path(p).is_file()},
            "result": dict(result),
            "seconds": round(seconds, 2),
            "warnings": list(warnings),
        }
        self.save()

    def warnings(self, stage: str) -> List[str]:
        entry = self.entries.get(stage)
        return list(entry.get("warnings", [])) if entry else []

    def seconds(self, stage: str) -> Optional[float]:
        entry = self.entries.get(stage)
        return entry.get("seconds") if entry else None

    def is_valid(self, stage: str, fp: str, outputs: Iterable[Path]) -> bool:
        entry = self.entries.get(stage)
        if not entry or entry.get("fingerprint") != fp:
            return False
        recorded: Mapping[str, int] = entry.get("outputs", {})
        for p in outputs:
            p = Path(p)
            if not p.is_file() or recorded.get(str(p)) != p.stat().st_size:
                return False
        return True

    def result(self, stage: str) -> Optional[Dict[str, Any]]:
        entry = self.entries.get(stage)
        return dict(entry["result"]) if entry else None

    def matches(self, stage: str, fp: str) -> bool:
        """Entry exists with this fingerprint (outputs may have been deleted on purpose)."""
        entry = self.entries.get(stage)
        return bool(entry and entry.get("fingerprint") == fp)
