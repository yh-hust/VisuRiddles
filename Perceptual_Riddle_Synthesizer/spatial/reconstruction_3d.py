from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spatial.render_core import draw_isometric_voxel, draw_matrix, font, make_connected_voxel, project_voxel, voxel_variants
from synth_engine.utils import ensure_dir

SUBRULE = "reconstruction_3d"


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
        top, front, side = project_voxel(voxel)
        candidates = voxel_variants(voxel, 4, rng)
        rng.shuffle(candidates)
        options = {label: cand for label, cand in zip("ABCD", candidates)}
        answer = next(label for label, cand in options.items() if cand == voxel)
        image_path = question_dir / f"puzzle_{qid}.png"
        _render_composite(image_path, top, front, side, options)
        record = {
            "id": qid,
            "rule": SUBRULE,
            "question": "Which 3D object matches the three orthographic views?",
            "correct_answer": answer,
            "voxel_matrix": voxel,
            "view_matrices": {"top": top, "front": front, "side": side},
            "source_json": str(question_dir / f"puzzle_{qid}.json"),
        }
        (question_dir / f"puzzle_{qid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    (out_path / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def _render_composite(output_path: Path, top: list[list[int]], front: list[list[int]], side: list[list[int]], options: dict[str, list[list[list[int]]]]) -> None:
    canvas = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "reconstruction 3d", fill="black", font=font(28))
    for idx, (label, matrix) in enumerate((("Top", top), ("Front", front), ("Side", side))):
        x = 110 + idx * 260
        draw.text((x, 95), label, fill="black", font=font(22))
        draw_matrix(draw, matrix, (x, 140), 46)
    draw.text((250, 380), "Which 3D object matches these views?", fill="black", font=font(22))
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
        draw_isometric_voxel(draw, options[label], (x + 88, top + 88), 18)
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
