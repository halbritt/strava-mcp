"""CLI entrypoint for strava-mcp."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from strava_mcp import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level strava-mcp argument parser."""
    parser = argparse.ArgumentParser(
        prog="strava-mcp",
        description="Read-only Strava MCP connector",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    parser.parse_args(list(argv) if argv is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
