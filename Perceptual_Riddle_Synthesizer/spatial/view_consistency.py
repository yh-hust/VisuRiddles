from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spatial.render_core import draw_isometric_voxel, draw_matrix, font, make_connected_voxel, matrix_variants, project_voxel
from synth_engine.utils import ensure_dir

SUBRULE = "view_consistency"


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
        _, _, side = project_voxel(voxel)
        opts = matrix_variants(side, 4, rng)
        rng.shuffle(opts)
        options = {label: opt for label, opt in zip("ABCD", opts)}
        answer = next(label for label, opt in options.items() if opt == side)
        image_path = question_dir / f"puzzle_{qid}.png"
        _render_composite(image_path, voxel, options)
        record = {
            "id": qid,
            "rule": SUBRULE,
            "question": "Choose the orthographic view that is consistent with the 3D object.",
            "correct_answer": answer,
            "voxel_matrix": voxel,
            "view_matrices": {"target": side, **{f"view_matrix_{k}": v for k, v in options.items()}},
            "source_json": str(question_dir / f"puzzle_{qid}.json"),
        }
        (question_dir / f"puzzle_{qid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    (out_path / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def _render_composite(output_path: Path, voxel: list[list[list[int]]], options: dict[str, list[list[int]]]) -> None:
    canvas = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "view consistency", fill="black", font=font(28))
    draw.text((110, 92), "3D object", fill="black", font=font(22))
    draw_isometric_voxel(draw, voxel, (280, 170), 38)
    draw.text((610, 92), "Target view", fill="black", font=font(22))
    draw.text((560, 125), "Choose the correct side view.", fill="black", font=font(18))
    draw.rectangle([620, 150, 760, 290], outline="black", width=3)
    draw.text((682, 192), "?", fill="black", font=font(48))
    _draw_option_panel(draw, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _draw_option_panel(draw: ImageDraw.ImageDraw, options: dict[str, list[list[int]]]) -> None:
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
        draw_matrix(draw, options[label], (x + 66, top + 34), 26, line_width=2)
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
