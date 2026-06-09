from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synth_engine.script_runner import run_category


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate positional data with paper-level or fine-grained subcategory counts.")
    parser.add_argument("--translate", type=int, default=0, help="translate subcategory count")
    parser.add_argument("--rotate", type=int, default=0, help="rotate subcategory count")
    parser.add_argument("--flip", type=int, default=0, help="flip subcategory count")
    parser.add_argument("--move_shift", type=int, default=0, help="fine-grained count for the fine-grained move_shift rule")
    parser.add_argument("--self_rotate", type=int, default=0, help="fine-grained count for the fine-grained self_rotate rule")
    parser.add_argument("--region_rotation", type=int, default=0, help="fine-grained count for the fine-grained region_rotation rule")
    parser.add_argument("--rotation_grid", type=int, default=0, help="fine-grained count for the fine-grained rotation_grid rule")
    parser.add_argument("--icon_count_rotation", type=int, default=0, help="fine-grained count for the fine-grained icon_count_rotation rule")
    parser.add_argument("--transform_3x3", type=int, default=0, help="fine-grained count for the fine-grained transform_3x3 rule")
    parser.add_argument("--mirror_flip", type=int, default=0, help="fine-grained count for the fine-grained mirror_flip rule")
    parser.add_argument("--flip_rotate_chain", type=int, default=0, help="fine-grained count for the fine-grained flip_rotate_chain rule")
    parser.add_argument("--out_root", type=str, default="./output_all", help="final data output root")
    parser.add_argument("--metadata_language", choices=["zh", "en"], default="zh", help="metadata language: zh or en")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    counts = {
        "translate": args.translate,
        "rotate": args.rotate,
        "flip": args.flip,
    }
    fine_counts = {
        "move_shift": args.move_shift,
        "self_rotate": args.self_rotate,
        "region_rotation": args.region_rotation,
        "rotation_grid": args.rotation_grid,
        "icon_count_rotation": args.icon_count_rotation,
        "transform_3x3": args.transform_3x3,
        "mirror_flip": args.mirror_flip,
        "flip_rotate_chain": args.flip_rotate_chain,
    }
    manifest = run_category("positional", counts, args.out_root, fine_subrule_counts=fine_counts, metadata_language=args.metadata_language)
    print(f"Generation complete. Final dataset: {manifest['output_root']}")


if __name__ == "__main__":
    main()
