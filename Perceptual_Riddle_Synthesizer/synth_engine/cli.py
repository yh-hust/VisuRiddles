from __future__ import annotations

import argparse
import json

from .config import load_config
from .logging_utils import setup_logging
from .orchestrator import EngineOrchestrator, validate_output
from .registry import DEFAULT_ORDER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified data synthesis engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_generate = subparsers.add_parser("generate", help="Run the synthesis engine")
    p_generate.add_argument("target", nargs="?", default="all", help="all or an internal module name")
    p_generate.add_argument("--config", type=str, default=None, help="Path to JSON/TOML/YAML config")
    p_generate.add_argument("--verbose", action="store_true")

    subparsers.add_parser("list-modules", help="List available internal modules")

    p_validate = subparsers.add_parser("validate", help="Validate a generated output directory")
    p_validate.add_argument("output_root", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-modules":
        print("\n".join(DEFAULT_ORDER))
        return 0

    if args.command == "validate":
        print(json.dumps(validate_output(args.output_root), ensure_ascii=False, indent=2))
        return 0

    setup_logging(args.verbose)
    config = load_config(args.config)
    selected = DEFAULT_ORDER if args.target == "all" else [args.target]
    manifest = EngineOrchestrator(config).run(selected)
    print(f"Generation complete. Final dataset: {manifest['output_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
