"""Command-line interface.

Usage::

    python -m floodplain_delineation.cli --config config/example.yaml

Exit status: 0 success; 1 expected failure (configuration, input, processing, output);
2 command-line usage error (argparse); 3 unexpected internal error.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .config import load_config
from .exceptions import FloodplainError
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="floodplain_delineation",
        description=(
            "Derive a preliminary potentially inundatable floodplain map from a DEM "
            "(HAND threshold). All parameters are read from a YAML file."
        ),
    )
    parser.add_argument("--config", "-c", required=True, metavar="FILE", help="Path to the YAML configuration file.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        summary = run_pipeline(config)
    except FloodplainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # unexpected: show type, keep non-zero status
        print(f"UNEXPECTED ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 3
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
