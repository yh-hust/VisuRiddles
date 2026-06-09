from __future__ import annotations

import copy
from pathlib import Path

from synth_engine.category_schema import CATEGORY_SCHEMA, POSITIONAL_FINE_SUBRULES
from synth_engine.config import DEFAULT_CONFIG, resolve_paths
from synth_engine.orchestrator import EngineOrchestrator
from synth_engine.utils import project_root


def build_category_config(
    category: str,
    subrule_counts: dict[str, int],
    out_root: str,
    fine_subrule_counts: dict[str, int] | None = None,
    metadata_language: str = "zh",
) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["output_root"] = out_root
    cfg["metadata_language"] = metadata_language
    for module_cfg in cfg["modules"].values():
        module_cfg["enabled"] = False
        module_cfg["count"] = 0
        if "subrule_counts" in module_cfg:
            module_cfg["subrule_counts"] = {key: 0 for key in module_cfg["subrule_counts"]}
        if "fine_grained_subrule_counts" in module_cfg:
            module_cfg["fine_grained_subrule_counts"] = {key: 0 for key in module_cfg["fine_grained_subrule_counts"]}
    module_cfg = cfg["modules"][category]
    counts = {name: max(0, int(subrule_counts.get(name, 0))) for name in CATEGORY_SCHEMA[category]}
    if category == "positional":
        fine_counts = {name: max(0, int((fine_subrule_counts or {}).get(name, 0))) for group in POSITIONAL_FINE_SUBRULES.values() for name in group}
        for category_name, group in POSITIONAL_FINE_SUBRULES.items():
            detailed_total = sum(fine_counts[name] for name in group)
            if detailed_total > 0:
                counts[category_name] += detailed_total
        module_cfg["fine_grained_subrule_counts"] = fine_counts
    module_cfg["enabled"] = sum(counts.values()) > 0
    module_cfg["count"] = sum(counts.values())
    module_cfg["subrule_counts"] = counts
    return resolve_paths(cfg, base_dir=project_root())


def run_category(
    category: str,
    subrule_counts: dict[str, int],
    out_root: str,
    fine_subrule_counts: dict[str, int] | None = None,
    metadata_language: str = "zh",
) -> dict:
    cfg = build_category_config(
        category,
        subrule_counts,
        out_root,
        fine_subrule_counts=fine_subrule_counts,
        metadata_language=metadata_language,
    )
    return EngineOrchestrator(cfg).run([category])
