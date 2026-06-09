from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from synth_engine.utils import ensure_dir, project_root

RAW_TYPES = ["five2one_symmetry_type", "three2three_symmetry_type"]


def generate(count: int, out_dir: str | Path, assets: dict[str, Any] | None = None) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    out_path = ensure_dir(out_dir)
    sym_icons = _image_pool("sym_images")
    non_sym_icons = _image_pool("not_sym_images") or sym_icons
    if len(sym_icons) < 8:
        raise RuntimeError("Attribute generation requires icon resources under resources/attribute/sym_images.")
    rng = random.Random(f"attribute:element:{total}")
    for idx in range(total):
        raw_type = RAW_TYPES[idx % len(RAW_TYPES)]
        title, choices, answer = _make_sequence(rng, sym_icons, non_sym_icons, stem_len=6)
        record = {"title": title, "choices": choices, "gt": answer, "type": raw_type}
        (out_path / f"element_{idx + 1}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def _image_pool(folder_name: str) -> list[Path]:
    folder = project_root() / "resources" / "attribute" / folder_name
    return sorted(folder.glob("*.png")) if folder.exists() else []


def _make_sequence(rng: random.Random, sym_icons: list[Path], non_sym_icons: list[Path], *, stem_len: int) -> tuple[list[str], list[str], str]:
    stem_icons = rng.sample(sym_icons, min(stem_len - 1, len(sym_icons)))
    title = [path.name for path in stem_icons] + ["question_mark.png"]
    choices = rng.sample(non_sym_icons + sym_icons, 4)
    answer_index = rng.randrange(4)
    if stem_icons:
        choices[answer_index] = stem_icons[0]
    return title, [path.name for path in choices], "ABCD"[answer_index]
