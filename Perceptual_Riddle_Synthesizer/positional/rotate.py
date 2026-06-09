from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from synth_engine.category_schema import split_count
from synth_engine.utils import ensure_dir, project_root

SUBRULE = "rotate"
FINE_GRAINED_SUBRULES = ["self_rotate", "region_rotation", "rotation_grid", "icon_count_rotation", "transform_3x3"]


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
    if len(icons) < 4:
        raise RuntimeError("Positional rotate generation requires at least four icons.")
    counts = _resolve_counts(total, subrule_counts)
    rng = random.Random(f"positional:{SUBRULE}:{counts}")
    question_index = 1
    for fine_rule in FINE_GRAINED_SUBRULES:
        for _ in range(counts.get(fine_rule, 0)):
            q_dir = ensure_dir(root / f"question{question_index}")
            if fine_rule == "self_rotate":
                _generate_self_rotate(q_dir, icons, qmark, rng)
            elif fine_rule == "region_rotation":
                _generate_region_rotation(q_dir, qmark, rng)
            elif fine_rule == "rotation_grid":
                _generate_rotation_grid(q_dir, icons, qmark, rng)
            elif fine_rule == "icon_count_rotation":
                _generate_icon_count_rotation(q_dir, icons, qmark, rng)
            else:
                _generate_transform_3x3(q_dir, icons, qmark, rng)
            question_index += 1
    return question_index - 1


def _resolve_counts(total: int, subrule_counts: dict[str, int] | None) -> dict[str, int]:
    provided = {name: max(0, int((subrule_counts or {}).get(name, 0))) for name in FINE_GRAINED_SUBRULES}
    if sum(provided.values()) <= 0:
        return split_count(total, FINE_GRAINED_SUBRULES)
    if sum(provided.values()) < total:
        extra = split_count(total - sum(provided.values()), FINE_GRAINED_SUBRULES)
        for name, value in extra.items():
            provided[name] += value
    return provided


def _generate_self_rotate(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    selected = rng.sample(icons, 4)
    direction = rng.choice(["clockwise", "counterclockwise"])
    self_angle = rng.choice([90, 180, 270])
    grids = []
    current = [_icon_state(p, 0) for p in selected]
    for _ in range(3):
        grids.append([item.copy() for item in current])
        current = _rotate_grid_states(current, direction, self_angle)
    answer_grid = _rotate_grid_states(current, direction, self_angle)
    for idx, grid in enumerate(grids, start=1):
        _render_icon_grid(grid).save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (220, 220)).save(q_dir / "4.png")
    options = [answer_grid] + _grid_state_distractors(answer_grid, rng, icons, 3)
    rng.shuffle(options)
    answer_idx = options.index(answer_grid)
    for label, grid in zip("ABCD", options):
        _render_icon_grid(grid).save(q_dir / f"{label}.png")
    meta = _base_meta("self_rotate", ["1.png", "2.png", "3.png", "4.png"], [1, 4], "ABCD"[answer_idx])
    meta["question_cells"] = [
        {"cell": f"{idx}.png", "grid_value": _grid_values(grid), "trans_rule": [{"trans_method": "rotate", "trans_direction": direction, "self_rotation": self_angle}]}
        for idx, grid in enumerate(grids, start=1)
    ] + [{"cell": "4.png"}]
    meta["ans_cells"] = [{"grid_value": _grid_values(answer_grid)}]
    _write_meta(q_dir, meta)


def _generate_region_rotation(q_dir: Path, qmark: Path, rng: random.Random) -> None:
    start = rng.sample(range(8), 2)
    direction = rng.choice(["clockwise", "counterclockwise"])
    step = rng.choice([1, 2])
    states = [_rotate_regions(start, direction, step * i) for i in range(4)]
    for idx, state in enumerate(states[:3], start=1):
        _render_region_square(state).save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (220, 220)).save(q_dir / "4.png")
    correct = states[3]
    distractors = _region_distractors(correct, rng)
    options = [correct] + distractors
    rng.shuffle(options)
    answer_idx = options.index(correct)
    for label, state in zip("ABCD", options):
        _render_region_square(state).save(q_dir / f"{label}.png")
    meta = _base_meta("region_rotation", ["1.png", "2.png", "3.png", "4.png"], [1, 4], "ABCD"[answer_idx])
    meta["question_cells"] = [
        {"cell": f"{idx}.png", "filled_regions": state, "trans_rule": [{"trans_method": "rotate_regions", "trans_direction": direction, "trans_step": step}]}
        for idx, state in enumerate(states[:3], start=1)
    ] + [{"cell": "4.png"}]
    meta["ans_cells"] = [{"filled_regions": correct}]
    _write_meta(q_dir, meta)


def _generate_rotation_grid(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    base_icon = _load_icon(rng.choice(icons), (200, 200))
    row_angles = [rng.choice([90, -90, 180]) for _ in range(3)]
    cells: list[Image.Image] = []
    row_sources = rng.sample(icons, 3)
    for row_idx, angle in enumerate(row_angles):
        current = _load_icon(row_sources[row_idx], (200, 200))
        for col in range(3):
            cells.append(current)
            current = current.rotate(angle, expand=False, fillcolor="white")
    correct_img = cells[8].rotate(row_angles[2], expand=False, fillcolor="white")
    for idx, img in enumerate(cells[:8], start=1):
        img.save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (200, 200)).save(q_dir / "9.png")
    distractors = [cells[8].rotate(a, expand=False, fillcolor="white") for a in [90, 180, 270, -90] if a != row_angles[2]]
    options = [correct_img] + distractors[:3]
    rng.shuffle(options)
    answer_idx = options.index(correct_img)
    for label, img in zip("ABCD", options):
        img.save(q_dir / f"{label}.png")
    meta = _base_meta("rotation_grid", [f"{i}.png" for i in range(1, 10)], [3, 3], "ABCD"[answer_idx])
    meta["question_cells"] = [
        {"cell": f"{idx}.png", "row": (idx - 1) // 3 + 1, "col": (idx - 1) % 3 + 1, "trans_rule": [{"trans_method": "row_rotation", "angle": row_angles[(idx - 1) // 3]}]}
        for idx in range(1, 10)
    ]
    meta["ans_cells"] = [{"row": 3, "col": 3, "angle": row_angles[2]}]
    _write_meta(q_dir, meta)


def _generate_icon_count_rotation(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    selected = rng.sample(icons, 4)
    direction = rng.choice(["clockwise", "counterclockwise"])
    counts = [rng.randint(1, 4) for _ in selected]
    states = []
    for step in range(4):
        order = _rotate_positions(list(range(4)), direction, step)
        state = []
        for pos in order:
            state.append((selected[pos], ((counts[pos] + step - 1) % 4) + 1))
        states.append(state)
    for idx, state in enumerate(states[:3], start=1):
        _render_count_grid(state).save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (240, 240)).save(q_dir / "4.png")
    correct = states[3]
    distractors = _count_distractors(correct, selected, rng)
    options = [correct] + distractors
    rng.shuffle(options)
    answer_idx = options.index(correct)
    for label, state in zip("ABCD", options):
        _render_count_grid(state).save(q_dir / f"{label}.png")
    meta = _base_meta("icon_count_rotation", ["1.png", "2.png", "3.png", "4.png"], [1, 4], "ABCD"[answer_idx])
    meta["question_cells"] = [
        {"cell": f"{idx}.png", "grid_value": [(p.name, n) for p, n in state], "trans_rule": [{"trans_method": "rotate_positions_and_counts", "trans_direction": direction}]}
        for idx, state in enumerate(states[:3], start=1)
    ] + [{"cell": "4.png"}]
    meta["ans_cells"] = [{"grid_value": [(p.name, n) for p, n in correct]}]
    _write_meta(q_dir, meta)


def _generate_transform_3x3(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    row_icons = rng.sample(icons, 3)
    row_rules = [rng.choice([("rotate", 90), ("rotate", 180), ("flip", "horizontal"), ("flip", "vertical")]) for _ in range(3)]
    cells: list[Image.Image] = []
    for icon_path, rule in zip(row_icons, row_rules):
        first = _load_icon(icon_path, (200, 200))
        second = _apply_image(first, *rule)
        third = _apply_image(second, *rule)
        cells.extend([first, second, third])
    correct = cells[8]
    for idx, img in enumerate(cells[:8], start=1):
        img.save(q_dir / f"{idx}.png")
    _load_qmark(qmark, (200, 200)).save(q_dir / "9.png")
    base_for_options = cells[7]
    correct = _apply_image(base_for_options, *row_rules[2])
    distractors = [_apply_image(base_for_options, op, value) for op, value in [("rotate", 90), ("rotate", 180), ("rotate", 270), ("flip", "horizontal"), ("flip", "vertical")] if (op, value) != row_rules[2]]
    options = [correct] + distractors[:3]
    rng.shuffle(options)
    answer_idx = options.index(correct)
    for label, img in zip("ABCD", options):
        img.save(q_dir / f"{label}.png")
    meta = _base_meta("transform_3x3", [f"{i}.png" for i in range(1, 10)], [3, 3], "ABCD"[answer_idx])
    meta["question_cells"] = [
        {"cell": f"{idx}.png", "row": (idx - 1) // 3 + 1, "col": (idx - 1) % 3 + 1, "trans_rule": [{"trans_method": row_rules[(idx - 1) // 3][0], "trans_value": row_rules[(idx - 1) // 3][1]}]}
        for idx in range(1, 10)
    ]
    meta["ans_cells"] = [{"row": 3, "col": 3, "trans_rule": row_rules[2]}]
    _write_meta(q_dir, meta)


def _base_meta(fine_rule: str, question_files: list[str], grid_type: list[int], answer: str) -> dict[str, Any]:
    return {"question_image": question_files, "rule_type": [SUBRULE], "subrule": fine_rule, "grid_type": grid_type, "answer": answer}


def _write_meta(q_dir: Path, meta: dict[str, Any]) -> None:
    (q_dir / "question.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _icon_state(path: Path, rotation: int = 0) -> dict[str, Any]:
    return {"path": path, "rotation": rotation}


def _rotate_grid_states(states: list[dict[str, Any]], direction: str, self_angle: int) -> list[dict[str, Any]]:
    order = [2, 0, 3, 1] if direction == "clockwise" else [1, 3, 0, 2]
    return [{"path": states[i]["path"], "rotation": (states[i]["rotation"] + self_angle) % 360} for i in order]


def _grid_state_distractors(answer: list[dict[str, Any]], rng: random.Random, icons: list[Path], count: int) -> list[list[dict[str, Any]]]:
    out = []
    while len(out) < count:
        variant = [item.copy() for item in answer]
        idx = rng.randrange(4)
        variant[idx]["rotation"] = (variant[idx]["rotation"] + rng.choice([90, 180, 270])) % 360
        if _grid_values(variant) != _grid_values(answer) and all(_grid_values(variant) != _grid_values(v) for v in out):
            out.append(variant)
    return out


def _grid_values(grid: list[dict[str, Any]]) -> list[list[Any]]:
    return [[item["path"].name, item["rotation"]] for item in grid]


def _render_icon_grid(states: list[dict[str, Any]], cell: int = 100, margin: int = 20) -> Image.Image:
    img = Image.new("RGB", (cell * 2 + margin * 2, cell * 2 + margin * 2), "white")
    draw = ImageDraw.Draw(img)
    for idx, item in enumerate(states):
        r, c = divmod(idx, 2)
        x0 = margin + c * cell
        y0 = margin + r * cell
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline="black", width=2)
        icon = _load_icon(item["path"], (cell - 14, cell - 14)).rotate(item["rotation"], expand=False, fillcolor="white")
        img.paste(icon.convert("RGB"), (x0 + 7, y0 + 7))
    return img


def _rotate_regions(indices: list[int], direction: str, step: int) -> list[int]:
    if direction == "clockwise":
        return sorted([(i + step) % 8 for i in indices])
    return sorted([(i - step) % 8 for i in indices])


def _render_region_square(indices: list[int], size: int = 220) -> Image.Image:
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = size // 2 - 20
    points = [(cx + radius * math_cos(k), cy + radius * math_sin(k)) for k in range(8)]
    for k in range(8):
        polygon = [(cx, cy), points[k], points[(k + 1) % 8]]
        draw.polygon(polygon, fill="black" if k in indices else "white", outline="black")
    draw.rectangle([20, 20, size - 20, size - 20], outline="black", width=3)
    return img


def math_cos(k: int) -> float:
    import math
    return math.cos(-math.pi / 2 + k * math.pi / 4)


def math_sin(k: int) -> float:
    import math
    return math.sin(-math.pi / 2 + k * math.pi / 4)


def _region_distractors(correct: list[int], rng: random.Random) -> list[list[int]]:
    out = []
    while len(out) < 3:
        state = sorted(rng.sample(range(8), len(correct)))
        if state != correct and state not in out:
            out.append(state)
    return out


def _rotate_positions(values: list[int], direction: str, step: int) -> list[int]:
    if direction == "clockwise":
        orders = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]]
    else:
        orders = [[0, 1, 2, 3], [1, 3, 0, 2], [3, 2, 1, 0], [2, 0, 3, 1]]
    return [values[i] for i in orders[step % 4]]


def _render_count_grid(state: list[tuple[Path, int]], cell: int = 110, margin: int = 20) -> Image.Image:
    img = Image.new("RGB", (cell * 2 + margin * 2, cell * 2 + margin * 2), "white")
    draw = ImageDraw.Draw(img)
    for idx, (path, count) in enumerate(state):
        r, c = divmod(idx, 2)
        x0 = margin + c * cell
        y0 = margin + r * cell
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], outline="black", width=2)
        icon = _load_icon(path, (30, 30))
        positions = [(x0 + 20, y0 + 20), (x0 + 60, y0 + 20), (x0 + 20, y0 + 60), (x0 + 60, y0 + 60)]
        for pos in positions[:count]:
            img.paste(icon.convert("RGB"), pos)
    return img


def _count_distractors(correct: list[tuple[Path, int]], icons: list[Path], rng: random.Random) -> list[list[tuple[Path, int]]]:
    out = []
    while len(out) < 3:
        variant = list(correct)
        idx = rng.randrange(4)
        path, count = variant[idx]
        variant[idx] = (path, (count % 4) + 1)
        if [(p.name, n) for p, n in variant] != [(p.name, n) for p, n in correct] and variant not in out:
            out.append(variant)
    return out


def _apply_image(img: Image.Image, op: str, value: Any) -> Image.Image:
    if op == "flip":
        if value == "horizontal":
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img.rotate(int(value), expand=False, fillcolor="white")


def _load_icon(path: Path, size: tuple[int, int] = (220, 220)) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", size, "white")
    img.thumbnail((size[0] - 24, size[1] - 24), Image.Resampling.LANCZOS)
    canvas.alpha_composite(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
    return canvas


def _load_qmark(path: Path, size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)


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
