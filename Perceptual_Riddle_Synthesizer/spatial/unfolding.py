from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spatial.render_core import draw_isometric_voxel, draw_net, font, make_connected_voxel, standard_nets
from synth_engine.utils import ensure_dir

SUBRULE = "unfolding"


def generate(count: int, out_dir: str | Path, *, seed: int = 42) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    out_path = ensure_dir(out_dir)
    records = _read_summary(out_path)
    rule_dir = ensure_dir(out_path / SUBRULE)
    rng = random.Random(f"spatial:{SUBRULE}:{seed}:{len(records)}")
    nets = standard_nets()
    for idx in range(1, total + 1):
        qid = f"{SUBRULE}_{idx}"
        question_dir = ensure_dir(rule_dir / f"puzzle_{qid}")
        voxel = make_connected_voxel(rng, 4, 6)
        net_choices = nets[:]
        rng.shuffle(net_choices)
        selected = net_choices[:4]
        answer_net = selected[0]
        rng.shuffle(selected)
        options = {label: net for label, net in zip("ABCD", selected)}
        answer = next(label for label, net in options.items() if net == answer_net)
        image_path = question_dir / f"puzzle_{qid}.png"
        _render_composite(image_path, voxel, options)
        record = {
            "id": qid,
            "rule": SUBRULE,
            "question": "Choose the cube net that matches the folded cube.",
            "correct_answer": answer,
            "voxel_matrix": voxel,
            "net_options": {k: v for k, v in options.items()},
            "source_json": str(question_dir / f"puzzle_{qid}.json"),
        }
        (question_dir / f"puzzle_{qid}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)
    (out_path / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return total


def _render_composite(output_path: Path, voxel: list[list[list[int]]], options: dict[str, list[tuple[int, int]]]) -> None:
    canvas = Image.new("RGB", (900, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 24), "unfolding", fill="black", font=font(28))
    draw.text((90, 90), "Folded cube", fill="black", font=font(22))
    draw_isometric_voxel(draw, voxel, (330, 170), 38)
    draw.text((590, 90), "Net = ?", fill="black", font=font(22))
    draw.rectangle([610, 170, 750, 310], outline="black", width=3)
    draw.text((670, 212), "?", fill="black", font=font(48))
    _draw_option_panel(draw, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _draw_option_panel(draw: ImageDraw.ImageDraw, options: dict[str, list[tuple[int, int]]]) -> None:
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
        draw_net(draw, options[label], (x + 48, top + 24), 24)
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
