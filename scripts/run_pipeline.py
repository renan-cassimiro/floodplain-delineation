#!/usr/bin/env python
"""Run the DEM-derived potential floodplain delineation pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from floodplain_delineation.config import load_config
from floodplain_delineation.exceptions import FloodplainError
from floodplain_delineation.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a DEM-derived potential floodplain map.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(args.config)
        run_pipeline(config)
    except FloodplainError as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 2
    except Exception as exc:  # Keep unexpected errors visible while preserving a useful CLI exit code.
        logging.exception("Unexpected error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
