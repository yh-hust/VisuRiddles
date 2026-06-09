from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from synth_engine.category_schema import split_count
from synth_engine.utils import ensure_dir, project_root

SUBRULE = "flip"
FINE_GRAINED_SUBRULES = ["mirror_flip", "flip_rotate_chain"]


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
        raise RuntimeError("Positional flip generation requires at least four icons.")
    counts = _resolve_counts(total, subrule_counts)
    rng = random.Random(f"positional:{SUBRULE}:{counts}")
    question_index = 1
    for fine_rule in FINE_GRAINED_SUBRULES:
        for _ in range(counts.get(fine_rule, 0)):
            q_dir = ensure_dir(root / f"question{question_index}")
            if fine_rule == "mirror_flip":
                _generate_mirror_flip(q_dir, icons, qmark, rng)
            else:
                _generate_flip_rotate_chain(q_dir, icons, qmark, rng)
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


def _generate_mirror_flip(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    icon1, icon2 = rng.sample(icons, 2)
    axis = rng.choice(["horizontal", "vertical"])
    base1 = _load_icon(icon1)
    base2 = _load_icon(icon2)
    stem = [base1, _flip(base1, axis), base2, _load_qmark(qmark, base1.size)]
    for idx, img in enumerate(stem, start=1):
        img.save(q_dir / f"{idx}.png")
    correct = _flip(base2, axis)
    distractors = _flip_distractors(base2, axis, rng)
    options = [correct] + distractors
    rng.shuffle(options)
    answer_idx = options.index(correct)
    for label, img in zip("ABCD", options):
        img.save(q_dir / f"{label}.png")
    meta = {
        "question_image": ["1.png", "2.png", "3.png", "4.png"],
        "rule_type": [SUBRULE],
        "subrule": "mirror_flip",
        "grid_type": [1, 4],
        "question_cells": [
            {"cell": "1.png", "grid_value": icon1.name, "trans_rule": []},
            {"cell": "2.png", "grid_value": icon1.name, "trans_rule": [{"trans_method": "flip", "trans_axis": axis}]},
            {"cell": "3.png", "grid_value": icon2.name, "trans_rule": []},
            {"cell": "4.png"},
        ],
        "ans_cells": [{"grid_value": icon2.name, "trans_rule": [{"trans_method": "flip", "trans_axis": axis}]}],
        "answer": "ABCD"[answer_idx],
    }
    (q_dir / "question.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_flip_rotate_chain(q_dir: Path, icons: list[Path], qmark: Path, rng: random.Random) -> None:
    icon1, icon2 = rng.sample(icons, 2)
    op1, val1 = rng.choice([("flip", "horizontal"), ("flip", "vertical"), ("rotate", 90), ("rotate", 180), ("rotate", 270)])
    op2, val2 = rng.choice([("flip", "horizontal"), ("flip", "vertical"), ("rotate", 90), ("rotate", 180), ("rotate", 270)])
    base1 = _load_icon(icon1)
    base2 = _load_icon(icon2)
    img2 = _apply(base1, op1, val1)
    img3 = _apply(img2, op2, val2)
    img5 = _apply(base2, op1, val1)
    answer = _apply(img5, op2, val2)
    stem = [base1, img2, img3, base2, img5, _load_qmark(qmark, base1.size)]
    for idx, img in enumerate(stem, start=1):
        img.save(q_dir / f"{idx}.png")
    options = [answer] + _operation_distractors(img5, op2, val2, rng)
    rng.shuffle(options)
    answer_idx = options.index(answer)
    for label, img in zip("ABCD", options):
        img.save(q_dir / f"{label}.png")
    meta = {
        "question_image": [f"{i}.png" for i in range(1, 7)],
        "rule_type": [SUBRULE],
        "subrule": "flip_rotate_chain",
        "grid_type": [2, 3],
        "question_cells": [
            {"cell": "1.png", "grid_value": icon1.name, "trans_rule": []},
            {"cell": "2.png", "grid_value": icon1.name, "trans_rule": [{"trans_method": op1, "trans_value": val1}]},
            {"cell": "3.png", "grid_value": icon1.name, "trans_rule": [{"trans_method": op1, "trans_value": val1}, {"trans_method": op2, "trans_value": val2}]},
            {"cell": "4.png", "grid_value": icon2.name, "trans_rule": []},
            {"cell": "5.png", "grid_value": icon2.name, "trans_rule": [{"trans_method": op1, "trans_value": val1}]},
            {"cell": "6.png"},
        ],
        "ans_cells": [{"grid_value": icon2.name, "trans_rule": [{"trans_method": op1, "trans_value": val1}, {"trans_method": op2, "trans_value": val2}]}],
        "answer": "ABCD"[answer_idx],
    }
    (q_dir / "question.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _flip(img: Image.Image, axis: str) -> Image.Image:
    return ImageOps.mirror(img) if axis == "horizontal" else ImageOps.flip(img)


def _apply(img: Image.Image, op: str, value: Any) -> Image.Image:
    if op == "flip":
        return _flip(img, str(value))
    return img.rotate(int(value), expand=False, fillcolor="white")


def _flip_distractors(base: Image.Image, correct_axis: str, rng: random.Random) -> list[Image.Image]:
    candidates = [base.copy(), _flip(base, "horizontal"), _flip(base, "vertical"), _flip(_flip(base, "horizontal"), "vertical")]
    correct = _flip(base, correct_axis)
    return _unique_images(candidates, correct, rng, 3)


def _operation_distractors(base: Image.Image, correct_op: str, correct_value: Any, rng: random.Random) -> list[Image.Image]:
    ops = [("flip", "horizontal"), ("flip", "vertical"), ("rotate", 90), ("rotate", 180), ("rotate", 270)]
    correct = _apply(base, correct_op, correct_value)
    candidates = [_apply(base, op, value) for op, value in ops if (op, value) != (correct_op, correct_value)]
    return _unique_images(candidates, correct, rng, 3)


def _unique_images(candidates: list[Image.Image], correct: Image.Image, rng: random.Random, count: int) -> list[Image.Image]:
    rng.shuffle(candidates)
    selected: list[Image.Image] = []
    signatures = {_signature(correct)}
    for candidate in candidates:
        sig = _signature(candidate)
        if sig not in signatures:
            signatures.add(sig)
            selected.append(candidate)
        if len(selected) == count:
            break
    while len(selected) < count:
        img = Image.new("RGBA", correct.size, "white")
        draw = ImageDraw.Draw(img)
        draw.ellipse([20 + 10 * len(selected), 20, 100, 100], outline="black", width=4)
        selected.append(img)
    return selected


def _signature(img: Image.Image) -> bytes:
    return img.convert("RGB").resize((32, 32)).tobytes()


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
