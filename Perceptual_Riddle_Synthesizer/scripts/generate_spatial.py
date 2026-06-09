from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from synth_engine.script_runner import run_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate spatial data with per-subcategory counts.")
    parser.add_argument("--unfolding", dest="unfolding", type=int, default=0, help="unfolding subcategory count")
    parser.add_argument("--three_view", dest="three_view", type=int, default=0, help="three_view subcategory count")
    parser.add_argument("--reconstruction_3d", dest="reconstruction_3d", type=int, default=0, help="reconstruction_3d subcategory count")
    parser.add_argument("--view_consistency", dest="view_consistency", type=int, default=0, help="view_consistency subcategory count")
    parser.add_argument("--multiple_views", dest="multiple_views", type=int, default=0, help="multiple_views subcategory count")
    parser.add_argument("--out_root", type=str, default="./output_all", help="final data output root")
    parser.add_argument("--metadata_language", choices=["zh", "en"], default="zh", help="metadata language: zh or en")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    counts = {
        "unfolding": args.unfolding,
        "three_view": args.three_view,
        "reconstruction_3d": args.reconstruction_3d,
        "view_consistency": args.view_consistency,
        "multiple_views": args.multiple_views,
    }
    manifest = run_category("spatial", counts, args.out_root, metadata_language=args.metadata_language)
    print(f"Generation complete. Final dataset: {manifest['output_root']}")


if __name__ == "__main__":
    main()
