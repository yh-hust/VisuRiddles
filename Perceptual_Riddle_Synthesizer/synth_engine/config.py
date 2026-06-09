from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from .utils import project_root

DEFAULT_CONFIG: dict[str, Any] = {
    "continue_on_error": True,
    "output_root": "outputs",
    "metadata_language": "zh",
    "format_normalization": {
        "enabled": True,
        "output_subdir": "",
        "image": {
            "cell_size": 160,
            "padding": 24,
            "gap": 18,
            "section_gap": 56,
            "border_width": 3,
            "label_height": 42,
        },
    },
    "modules": {
        "attribute": {
            "enabled": False,
            "count": 0,
            "output_subdir": "attribute",
            "subrule_counts": {"element": 0, "group": 0},
            "assets": {
                "font_path": None,
                "sym_img_root": "resources/attribute/sym_images",
                "other_img_root": "resources",
                "sym_json_path": "resources/attribute/mock_data.json",
                "not_sym_img_root": "resources/attribute/not_sym_images",
            },
        },
        "positional": {
            "enabled": True,
            "count": 10,
            "output_subdir": "positional",
            "subrule_counts": {"translate": 0, "rotate": 0, "flip": 0},
            "fine_grained_subrule_counts": {
                "move_shift": 0,
                "self_rotate": 0,
                "region_rotation": 0,
                "rotation_grid": 0,
                "icon_count_rotation": 0,
                "transform_3x3": 0,
                "mirror_flip": 0,
                "flip_rotate_chain": 0
            },
            "resources": {
                "icon_dir": "resources/positional/icon",
                "icon2_dir": "resources/positional/icon2",
                "shape_dir": "resources/positional/complex_shapes",
                "question_mark": "resources/question_mark.png",
            },
        },
        "spatial": {
            "enabled": True,
            "count": 10,
            "output_subdir": "spatial",
            "subrule_counts": {
                "unfolding": 0,
                "three_view": 0,
                "reconstruction_3d": 0,
                "view_consistency": 0,
                "multiple_views": 0,
            },
            "resource_root": "resources/spatial",
            "seed": 42,
        },
        "numerical": {
            "enabled": True,
            "count": 100,
            "output_subdir": "numerical",
            "subrule_counts": {"line": 0, "curve": 0, "angle": 0, "cart": 0, "space": 0, "parts": 0},
            "input_json": "resources/numerical/merge_data.json",
            "output_filename": "questions.json",
            "qimage": "resources/question_mark.png",
        },
        "stylistic": {
            "enabled": True,
            "count": 100,
            "output_subdir": "stylistic",
            "subrule_counts": {"and": 0, "or": 0, "xor": 0, "xnor": 0},
        },
        "raven": {
            "enabled": True,
            "count": 20,
            "output_subdir": "raven",
            "subrule_counts": {"raven": 20},
            "dataset_json": "resources/raven/test1.json",
            "seed": 42,
        },
        "sudoku": {
            "enabled": True,
            "count": 20,
            "output_subdir": "sudoku",
            "subrule_counts": {"sudoku": 20},
            "levels": [1, 2, 3, 4, 5],
            "seed": 42,
        },
    },
}


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_toml(path: Path) -> dict[str, Any]:
    import tomllib
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("YAML config requires PyYAML to be installed.") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return resolve_paths(cfg, base_dir=project_root())
    path = Path(config_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    elif suffix == ".toml":
        loaded = _load_toml(path)
    elif suffix in {".yaml", ".yml"}:
        loaded = _load_yaml(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    _deep_update(cfg, loaded)
    return resolve_paths(cfg, base_dir=path.parent)


def _resolve_maybe_path(base_dir: Path, value: Any, *, prefer_existing_project_path: bool = True) -> Any:
    if not isinstance(value, str) or value == "":
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    base_candidate = (base_dir / p).resolve()
    project_candidate = (project_root() / p).resolve()
    if prefer_existing_project_path and project_candidate.exists() and not base_candidate.exists():
        return str(project_candidate)
    return str(base_candidate)


PATH_KEYS = {
    "output_root",
    "lib_dir",
    "input_json",
    "qimage",
    "font_path",
    "sym_img_root",
    "other_img_root",
    "sym_json_path",
    "not_sym_img_root",
    "dataset_json",
    "resource_root",
    "icon_dir",
    "icon2_dir",
    "shape_dir",
    "question_mark",
}


def _resolve_path_values(container: dict[str, Any], base_dir: Path) -> None:
    for key, value in list(container.items()):
        if key in PATH_KEYS:
            container[key] = _resolve_maybe_path(base_dir, value)
        elif isinstance(value, dict):
            _resolve_path_values(value, base_dir)


def resolve_paths(cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if "output_root" in cfg:
        cfg["output_root"] = _resolve_maybe_path(base_dir, cfg["output_root"], prefer_existing_project_path=False)
    fmt_cfg = cfg.get("format_normalization")
    if isinstance(fmt_cfg, dict):
        _resolve_path_values(fmt_cfg, base_dir)
    for module_cfg in cfg.get("modules", {}).values():
        _resolve_path_values(module_cfg, base_dir)
    return cfg
