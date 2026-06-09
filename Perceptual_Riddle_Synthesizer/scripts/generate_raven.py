from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from synth_engine.script_runner import run_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate raven data with per-subcategory counts.")
    parser.add_argument("--raven", dest="raven", type=int, default=0, help="raven subcategory count")
    parser.add_argument("--out_root", type=str, default="./output_all", help="final data output root")
    parser.add_argument("--metadata_language", choices=["zh", "en"], default="zh", help="metadata language: zh or en")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    counts = {
        "raven": args.raven,
    }
    manifest = run_category("raven", counts, args.out_root, metadata_language=args.metadata_language)
    print(f"Generation complete. Final dataset: {manifest['output_root']}")


if __name__ == "__main__":
    main()
