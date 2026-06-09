from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from synth_engine.utils import ensure_dir, project_root

RULE_DETAILS = {
    "line": {"feature": "line", "patterns": ["increasing_1", "decreasing_1", "constant"]},
    "curve": {"feature": "curve", "patterns": ["increasing_2", "decreasing_2", "constant"]},
    "angle": {"feature": "angle", "patterns": ["increasing_3", "decreasing_3", "constant"]},
    "cart": {"feature": "cart", "patterns": ["always_even", "always_odd"]},
    "space": {"feature": "space", "patterns": ["even_odd_even_odd", "odd_even_odd_even"]},
    "parts": {"feature": "parts", "patterns": ["increasing_1", "decreasing_1", "always_even", "always_odd"]},
}


def generate_subrule(subrule: str, count: int, out_dir: str | Path, input_json: str | None = None, qimage: str | None = None) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    if subrule not in RULE_DETAILS:
        raise ValueError(f"Unknown numerical subrule: {subrule}")

    out_path = ensure_dir(Path(out_dir) / subrule)
    question_mark = _resolve_question_mark(qimage)
    icon_pool = _icon_pool()
    if len(icon_pool) < 12:
        raise RuntimeError("Numerical generation requires at least 12 icon images in resources/numerical.")

    rng = random.Random(f"numerical:{subrule}:{total}")
    records: list[dict[str, Any]] = []
    details = RULE_DETAILS[subrule]
    patterns = details["patterns"]

    for idx in range(1, total + 1):
        pattern = patterns[(idx - 1) % len(patterns)]
        stem_icons = rng.sample(icon_pool, 4)
        option_icons = rng.sample([p for p in icon_pool if p not in stem_icons], 4)
        answer_label = "ABCD"[(idx - 1) % 4]
        answer_index = "ABCD".index(answer_label)
        option_features = _option_features(pattern, answer_index)
        stem_features = _stem_features(pattern)

        record = {
            "id": str(idx),
            "type": "numerical_pattern",
            "canonical_rule": subrule,
            "feature": details["feature"],
            "rule": pattern,
            "grid": [5, 1, 1],
            "cells": [
                {"cell": str(path), "feature": stem_features[i]} for i, path in enumerate(stem_icons)
            ] + [{"cell": str(question_mark), "feature": None}],
            "answer_cells": [
                {label: str(option_icons[i]), "feature": option_features[i]} for i, label in enumerate("ABCD")
            ],
            "answer": answer_label,
            "description": _description(subrule, pattern),
        }
        _render_preview(out_path / f"question_{idx}.png", stem_icons, question_mark, option_icons)
        records.append(record)

    (out_path / "questions.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def _resolve_question_mark(qimage: str | None) -> Path:
    candidates = [Path(qimage)] if qimage else []
    candidates.append(project_root() / "resources" / "question_mark.png")
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError("resources/question_mark.png not found")


def _icon_pool() -> list[Path]:
    root = project_root() / "resources" / "numerical"
    images = []
    for subdir in ("NewIcons", "OutputIcons"):
        folder = root / subdir
        if folder.exists():
            images.extend(sorted(folder.glob("*.png")))
    if not images:
        images.extend(sorted(root.rglob("*.png")))
    return images


def _stem_features(pattern: str) -> list[int]:
    if pattern.startswith("increasing"):
        step = int(pattern.rsplit("_", 1)[-1]) if "_" in pattern else 1
        return [1 + i * step for i in range(4)]
    if pattern.startswith("decreasing"):
        step = int(pattern.rsplit("_", 1)[-1]) if "_" in pattern else 1
        return [9 - i * step for i in range(4)]
    if pattern == "always_even":
        return [2, 4, 6, 8]
    if pattern == "always_odd":
        return [1, 3, 5, 7]
    if pattern == "even_odd_even_odd":
        return [2, 3, 4, 5]
    if pattern == "odd_even_odd_even":
        return [1, 2, 3, 4]
    return [5, 5, 5, 5]


def _option_features(pattern: str, answer_index: int) -> list[int]:
    stem = _stem_features(pattern)
    if pattern.startswith("increasing"):
        step = int(pattern.rsplit("_", 1)[-1]) if "_" in pattern else 1
        correct = stem[-1] + step
    elif pattern.startswith("decreasing"):
        step = int(pattern.rsplit("_", 1)[-1]) if "_" in pattern else 1
        correct = stem[-1] - step
    elif pattern == "always_even":
        correct = 10
    elif pattern == "always_odd":
        correct = 9
    elif pattern == "even_odd_even_odd":
        correct = 6
    elif pattern == "odd_even_odd_even":
        correct = 5
    else:
        correct = stem[-1]
    values = [correct + delta for delta in (-2, -1, 1, 2)]
    values[answer_index] = correct
    return values


def _description(subrule: str, pattern: str) -> str:
    return f"The {subrule} feature follows the {pattern} numerical rule."


def _render_preview(output_path: Path, stem_icons: list[Path], qmark: Path, option_icons: list[Path]) -> None:
    cell = 120
    gap = 18
    width = 5 * cell + 4 * gap
    height = cell * 2 + 92
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(stem_icons + [qmark]):
        _paste_icon(canvas, path, idx * (cell + gap), 0, cell)
    opt_y = cell + 56
    for idx, path in enumerate(option_icons):
        x = idx * (cell + gap)
        _paste_icon(canvas, path, x, opt_y, cell)
        draw.text((x + cell // 2 - 5, opt_y + cell + 8), "ABCD"[idx], fill="black")
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def _paste_icon(canvas: Image.Image, path: Path, x: int, y: int, cell: int) -> None:
    with Image.open(path) as im:
        icon = im.convert("RGBA")
        icon.thumbnail((cell - 24, cell - 24))
        ox = x + (cell - icon.width) // 2
        oy = y + (cell - icon.height) // 2
        canvas.paste(icon.convert("RGB"), (ox, oy))


def generate(count: int, out_dir: str | Path, input_json: str | None = None, qimage: str | None = None) -> int:
    return generate_subrule('cart', count, out_dir, input_json, qimage)
