from __future__ import annotations

import copy
from argparse import Namespace

from .category_schema import CATEGORY_ORDER, CATEGORY_SCHEMA, split_count
from .config import DEFAULT_CONFIG, resolve_paths
from .utils import project_root

PUBLIC_CATEGORY_ARGS = CATEGORY_ORDER


def build_generate_config(args: Namespace) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["output_root"] = args.out_root
    cfg["metadata_language"] = getattr(args, "metadata_language", "zh")
    for category in CATEGORY_ORDER:
        count = max(0, int(getattr(args, category)))
        module_cfg = cfg["modules"][category]
        module_cfg["enabled"] = count > 0
        module_cfg["count"] = count
        module_cfg["subrule_counts"] = split_count(count, CATEGORY_SCHEMA[category])
    return resolve_paths(cfg, base_dir=project_root())
