"""Stage executor: runs the heavy stages in dependency order, skipping up-to-date ones on resume.

A stage is *up to date* when its manifest fingerprint matches and its output files exist with the
recorded size (see :mod:`manifest`). ``ensure(name)`` first makes sure the stage's dependencies are
available (running them only if they are not up to date), then runs the stage itself. When a
downstream stage is already up to date, its upstream stages are not touched at all, so e.g. changing
only the HAND threshold, or deleting an unwanted intermediate, never forces a recomputation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import resources
from .logging_utils import WarningCollector, get_logger
from .manifest import StageManifest

log = get_logger("stages")


@dataclass
class StageSpec:
    name: str
    label: str
    outputs: Sequence[Path]
    fingerprint: str
    deps: Sequence[str]
    run: Callable[[], Dict[str, Any]]


@dataclass
class StageExecutor:
    manifest: StageManifest
    warnings: WarningCollector
    resume: bool
    specs: Dict[str, StageSpec] = field(default_factory=dict)
    done: set = field(default_factory=set)
    reused: List[str] = field(default_factory=list)
    timings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_peak: Optional[Callable[[], Optional[int]]] = None   # highest sampled WhiteboxTools memory

    def add(self, spec: StageSpec) -> None:
        self.specs[spec.name] = spec

    # ------------------------------------------------------------------ #
    def ensure(self, name: str) -> None:
        if name in self.done:
            return
        spec = self.specs[name]
        if self.resume and self.manifest.is_valid(name, spec.fingerprint, spec.outputs):
            log.info(
                "Stage '%s' is up to date (same inputs and parameters as the previous run); skipped.", spec.label,
            )
            self.done.add(name)
            return
        for dep in spec.deps:
            self.ensure(dep)
        self.manifest.invalidate(name)  # a crash while running must not leave a stale 'valid' entry
        log.info("Stage '%s': started.", spec.label)
        n_warn = len(self.warnings)
        started = time.monotonic()
        result = spec.run()
        seconds = time.monotonic() - started
        self.manifest.record(
            name, spec.fingerprint, spec.outputs, result, seconds,
            warnings=self.warnings.messages[n_warn:],
        )
        self.timings[name] = {"status": "run", "seconds": round(seconds, 2)}
        self.done.add(name)
        log.info(
            "Stage '%s': finished in %s (Python peak memory so far %s; WhiteboxTools peak sampled so far %s).",
            spec.label, resources.format_duration(seconds),
            resources.format_bytes(resources.self_peak_rss_bytes()),
            resources.format_bytes(self.tool_peak() if self.tool_peak else None),
        )

    def collect_reused(self, order: Sequence[str]) -> List[str]:
        """After ``ensure``: stages that were not run now but have current results from a previous run.

        Their warnings are added again so the processing summary stays complete. Stages that were never
        consulted (because a downstream stage was already up to date) are included.
        """
        self.reused = []
        for name in order:
            spec = self.specs[name]
            if self.timings.get(name, {}).get("status") == "run" or not self.manifest.matches(name, spec.fingerprint):
                continue
            self.reused.append(name)
            self.timings[name] = {"status": "reused", "original_seconds": self.manifest.seconds(name)}
            for message in self.manifest.warnings(name):
                self.warnings.add(f"{message} [from a previous run of stage '{name}']")
        return self.reused

    def result(self, name: str) -> Optional[Dict[str, Any]]:
        """Result of a stage from this or a previous run, only if its fingerprint is current."""
        spec = self.specs.get(name)
        if spec is None or not self.manifest.matches(name, spec.fingerprint):
            return None
        return self.manifest.result(name)
