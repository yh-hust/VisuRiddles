from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spatial.render_core import draw_isometric_voxel, font, make_connected_voxel, rotate_z, voxel_variants
from synth_engine.utils import ensure_dir

SUBRULE = "multiple_views"


def generate(count: int, out_dir: str | Path, *, seed: int = 42) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    out_path = ensure_dir(out_dir)
    records = _read_summary(out_path)
    rule_dir = ensure_dir(out_path / SUBRULE)
    rng = random.Random(f"spatial:{SUBRULE}:{seed}:{len(records)}")
    for idx in range(1, total + 1):
        qid = f"{SUBRULE}_{idx}"
        question_dir = ensure_dir(rule_dir / f"puzzle_{qid}")
        voxel = make_connected_voxel(rng)
        rotated = rotate_z(voxel)
        candidates = voxel_variants(voxel, 4, rng)
        rng.shuffle(candidates)
        options = {label: cand for label, cand in zip("ABCD", candidates)}
        answer = next(label for label, cand in options.items() if cand == voxel)
        image_path = question_dir / f"puzzle_{qid}.png"
        _render_composite(image_path, voxel, rotated, options)
        record = {
            "id": qid,
            "rule": SUBRULE,
            "question": "Choose another spatial view that is consistent with the observed object.",
            "correct_answer": answer,
            "voxel_matrix": voxel,
            "source_json": str(question_dir / f"puzzle_{qid}.json"),
        }
        (question_dir / f"puzzle_{qid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    (out_path / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def _render_composite(output_path: Path, voxel: list[list[list[int]]], rotated: list[list[list[int]]], options: dict[str, list[list[list[int]]]]) -> None:
    canvas = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "multiple views", fill="black", font=font(28))
    draw.text((100, 92), "View 1", fill="black", font=font(22))
    draw_isometric_voxel(draw, voxel, (240, 180), 32)
    draw.text((470, 92), "View 2", fill="black", font=font(22))
    draw_isometric_voxel(draw, rotated, (600, 180), 32)
    draw.text((250, 390), "Choose another consistent view", fill="black", font=font(22))
    _draw_option_panel(draw, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _draw_option_panel(draw: ImageDraw.ImageDraw, options: dict[str, list[list[list[int]]]]) -> None:
    top = 500
    left = 30
    panel_w = 840
    cell_w = 210
    box_h = 150
    draw.rectangle([left, top, left + panel_w, top + box_h], outline="black", width=3)
    for idx, label in enumerate("ABCD"):
        x = left + idx * cell_w
        if idx > 0:
            draw.line((x, top, x, top + box_h), fill="black", width=3)
        draw_isometric_voxel(draw, options[label], (x + 88, top + 90), 18)
        draw.text((x + cell_w // 2 - 9, top + box_h + 10), label, fill="black", font=font(22))


def _read_summary(out_path: Path) -> list[dict[str, Any]]:
    summary = out_path / "summary.json"
    if not summary.exists():
        return []
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []
