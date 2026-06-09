from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from synth_engine.category_schema import split_count
from synth_engine.utils import ensure_dir, project_root

SUBRULE = "translate"
FINE_GRAINED_SUBRULES = ["move_shift"]


def generate(
    count: int,
    out_dir: str | Path,
    resources: dict[str, Any] | None = None,
    subrule_counts: dict[str, int] | None = None,
) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    root = Path(out_dir) / SUBRULE
    icons = _icon_pool(resources or {})
    qmark = _question_mark(resources or {})
    if not icons:
        raise RuntimeError("Positional generation requires icon resources under resources/positional.")
    counts = _resolve_counts(total, subrule_counts)
    rng = random.Random(f"positional:{SUBRULE}:{counts}")
    question_index = 1
    for fine_rule in FINE_GRAINED_SUBRULES:
        for _ in range(counts.get(fine_rule, 0)):
            q_dir = ensure_dir(root / f"question{question_index}")
            _generate_move_shift(q_dir, icons, qmark, rng)
            question_index += 1
    return question_index - 1


def _resolve_counts(total: int, subrule_counts: dict[str, int] | None) -> dict[str, int]:
    provided = {name: max(0, int((subrule_counts or {}).get(name, 0))) for name in FINE_GRAINED_SUBRULES}
    if sum(provided.values()) <= 0:
        return split_count(total, FINE_GRAINED_SUBRULES)
    if sum(provided.values()) < total:
        provided[FINE_GRAINED_SUBRULES[0]] += total - sum(provided.values())
    return provided


def _generate_move_shift(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    icon = rng.choice(icons)
    size = rng.choice([3, 4])
    direction = rng.choice(["right", "left", "down", "up"])
    step = rng.choice([1, 2])
    start = (rng.randrange(size), rng.randrange(size))
    positions = [_shift(start, direction, step * t, size) for t in range(4)]
    question_positions = positions[:3]
    answer_pos = positions[3]
    for idx, pos in enumerate(question_positions, start=1):
        _render_position_grid(size, pos, icon).save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (220, 220)).save(q_dir / "4.png")
    distractor_positions = _distractor_positions(answer_pos, size, rng, count=3)
    option_positions = [answer_pos] + distractor_positions
    rng.shuffle(option_positions)
    answer_idx = option_positions.index(answer_pos)
    for label, pos in zip("ABCD", option_positions):
        _render_position_grid(size, pos, icon).save(q_dir / f"{label}.png")
    meta = {
        "question_image": ["1.png", "2.png", "3.png", "4.png"],
        "rule_type": [SUBRULE],
        "subrule": "move_shift",
        "grid_type": [1, 4],
        "question_cells": [
            {
                "cell": f"{idx}.png",
                "grid_size": [size, size],
                "icon": icon.name,
                "position": list(pos),
                "trans_rule": [{"trans_method": "translate", "trans_direction": direction, "trans_step": step}],
            }
            for idx, pos in enumerate(question_positions, start=1)
        ] + [{"cell": "4.png"}],
        "ans_cells": [{"grid_size": [size, size], "position": list(answer_pos), "icon": icon.name}],
        "answer": "ABCD"[answer_idx],
    }
    (q_dir / "question.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _shift(pos: tuple[int, int], direction: str, step: int, size: int) -> tuple[int, int]:
    r, c = pos
    if direction == "right":
        return r, (c + step) % size
    if direction == "left":
        return r, (c - step) % size
    if direction == "down":
        return (r + step) % size, c
    return (r - step) % size, c


def _distractor_positions(answer: tuple[int, int], size: int, rng: random.Random, count: int) -> list[tuple[int, int]]:
    pool = [(r, c) for r in range(size) for c in range(size) if (r, c) != answer]
    return rng.sample(pool, count)


def _render_position_grid(size: int, position: tuple[int, int], icon_path: Path, cell: int = 58, margin: int = 12) -> Image.Image:
    canvas_size = size * cell + margin * 2
    img = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)
    for r in range(size):
        for c in range(size):
            x0 = margin + c * cell
            y0 = margin + r * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline="black", width=2)
    icon = Image.open(icon_path).convert("RGBA")
    icon.thumbnail((cell - 14, cell - 14), Image.Resampling.LANCZOS)
    r, c = position
    x = margin + c * cell + (cell - icon.width) // 2
    y = margin + r * cell + (cell - icon.height) // 2
    img.paste(icon, (x, y), icon)
    return img


def _icon_pool(resources: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for key in ("icon_dir", "icon2_dir", "shape_dir"):
        if resources.get(key):
            roots.append(Path(resources[key]))
    roots.append(project_root() / "resources" / "positional")
    images: list[Path] = []
    for root in roots:
        if root.exists():
            images.extend(sorted(p for p in root.rglob("*.png") if p.name != "question_mark.png"))
    return images


def _question_mark(resources: dict[str, Any]) -> Path:
    for value in (resources.get("question_mark"), project_root() / "resources" / "question_mark.png"):
        path = Path(value) if value else None
        if path and path.exists():
            return path
    raise FileNotFoundError("resources/question_mark.png not found")


def _load_qmark(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
