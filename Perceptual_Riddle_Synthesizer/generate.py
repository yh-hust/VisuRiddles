from __future__ import annotations

import argparse

from synth_engine import generate_config
from synth_engine.orchestrator import EngineOrchestrator

PUBLIC_CATEGORY_ARGS = generate_config.PUBLIC_CATEGORY_ARGS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all seven canonical puzzle categories. "
            "The first seven parameters are category-level question counts."
        )
    )
    parser.add_argument("--attribute", type=int, default=0, help="attribute category total count")
    parser.add_argument("--positional", type=int, default=10, help="positional category total count")
    parser.add_argument("--spatial", type=int, default=10, help="spatial category total count")
    parser.add_argument("--numerical", type=int, default=100, help="numerical category total count")
    parser.add_argument("--stylistic", type=int, default=100, help="stylistic category total count")
    parser.add_argument("--raven", type=int, default=20, help="raven category total count")
    parser.add_argument("--sudoku", type=int, default=20, help="sudoku category total count")
    parser.add_argument("--out_root", type=str, default="./output_all", help="final data output root")
    parser.add_argument("--metadata_language", choices=["zh", "en"], default="zh", help="metadata language: zh or en")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = generate_config.build_generate_config(args)
    manifest = EngineOrchestrator(config).run()
    print(f"Generation complete. Final dataset: {manifest['output_root']}")


if __name__ == "__main__":
    main()
