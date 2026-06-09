from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ANGLE_RE = re.compile(r'旋转(-?\d+)度')
SIZE_RE = re.compile(r'(小号|小中号|中号|大中号|大号)')
COLOR_RE = re.compile(r'(黑色|深灰色|浅灰色|亮白色|白色)')
SHAPE_RE = re.compile(r'(圆形|三角形|四边形|五边形|六边形)')

SIZE_SCALE = {'小号': 0.26, '小中号': 0.34, '中号': 0.42, '大中号': 0.50, '大号': 0.58}
COLOR_FILL = {'黑色': (20, 20, 20), '深灰色': (80, 80, 80), '浅灰色': (170, 170, 170), '亮白色': (245, 245, 245), '白色': (255, 255, 255)}


def _font(size: int):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


def _regular_polygon(cx: float, cy: float, radius: float, sides: int, rotation_deg: float) -> list[tuple[float, float]]:
    pts = []
    start = math.radians(rotation_deg - 90)
    for i in range(sides):
        theta = start + i * 2 * math.pi / sides
        pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return pts


def _parse_cell(desc: str) -> dict[str, Any]:
    desc = desc.strip().rstrip('。')
    if not desc or desc == '空':
        return {'empty': True, 'raw': desc}
    angle = int(ANGLE_RE.search(desc).group(1)) if ANGLE_RE.search(desc) else 0
    size = SIZE_RE.search(desc).group(1) if SIZE_RE.search(desc) else '中号'
    color = COLOR_RE.search(desc).group(1) if COLOR_RE.search(desc) else '黑色'
    shape = SHAPE_RE.search(desc).group(1) if SHAPE_RE.search(desc) else None
    return {'empty': False, 'angle': angle, 'size': size, 'color': color, 'shape': shape, 'raw': desc}


def _parse_panel(text: str, expected: int) -> list[dict[str, Any]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cells = []
    for ln in lines:
        try:
            content = ln.split('：', 1)[1]
        except IndexError:
            content = ln
        cells.append(_parse_cell(content))
    if len(cells) != expected:
        raise ValueError(f'Expected {expected} cells, got {len(cells)}')
    return cells


def _render_cell(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], spec: dict[str, Any]) -> None:
    x0, y0, x1, y1 = bbox
    draw.rectangle(bbox, outline=(180, 180, 180), width=1)
    if spec.get('empty'):
        return
    if spec.get('shape') is None:
        text = spec.get('raw', '?')[:4]
        font = _font(14)
        tb = draw.textbbox((0, 0), text, font=font)
        draw.text(((x0 + x1 - (tb[2]-tb[0]))/2, (y0 + y1 - (tb[3]-tb[1]))/2), text, fill='black', font=font)
        return
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    radius = min(x1 - x0, y1 - y0) * SIZE_SCALE.get(spec['size'], 0.42)
    fill = COLOR_FILL.get(spec['color'], (100, 100, 100))
    outline = (30, 30, 30)
    shape = spec['shape']
    angle = spec['angle']
    if shape == '圆形':
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=fill, outline=outline, width=2)
    else:
        sides = {'三角形': 3, '四边形': 4, '五边形': 5, '六边形': 6}[shape]
        draw.polygon(_regular_polygon(cx, cy, radius, sides, angle), fill=fill, outline=outline)


def _render_grid(specs: list[dict[str, Any]], rows: int, cols: int, cell: int = 92, padding: int = 12, title: str | None = None) -> Image.Image:
    title_h = 34 if title else 0
    w = cols * cell + (cols + 1) * padding
    h = rows * cell + (rows + 1) * padding + title_h
    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)
    if title:
        draw.text((padding, 6), title, fill='black', font=_font(20))
    for idx, spec in enumerate(specs):
        r = idx // cols
        c = idx % cols
        x0 = padding + c * (cell + padding)
        y0 = title_h + padding + r * (cell + padding)
        _render_cell(draw, (x0, y0, x0 + cell, y0 + cell), spec)
    return img


def _stack_vertical(images: list[Image.Image], gap: int = 16, labels: list[str] | None = None) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    total_h = sum(heights) + gap * (len(images) - 1)
    out = Image.new('RGB', (max(widths), total_h), 'white')
    y = 0
    for img in images:
        out.paste(img, ((out.width - img.width)//2, y))
        y += img.height + gap
    return out


def generate_raven_questions(count: int, out_dir: str | Path, dataset_json: str | Path | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = Path(dataset_json or (Path(__file__).resolve().parent / 'test1.json'))
    data = json.loads(dataset_json.read_text(encoding='utf-8'))
    rng = random.Random(seed)
    samples = [rng.choice(data) for _ in range(count)]
    records = []
    for idx, item in enumerate(samples):
        content = _parse_panel(item['content'], 9)
        choices = _parse_panel(item['choices'], 8)
        content_img = _render_grid(content, 3, 3, title='RAVEN Pattern')
        choices_img = _render_grid(choices, 2, 4, title='Choices')
        combined = _stack_vertical([content_img, choices_img])
        combined_path = out_dir / f'raven_{idx}.png'
        choices_path = out_dir / f'raven_{idx}_choices.png'
        content_path = out_dir / f'raven_{idx}_content.png'
        content_img.save(content_path)
        choices_img.save(choices_path)
        combined.save(combined_path)
        records.append({
            'id': idx,
            'image_path': str(combined_path),
            'content_image_path': str(content_path),
            'choices_image_path': str(choices_path),
            'content': item['content'],
            'choices': item['choices'],
            'answer': int(item['answer']),
            'source_dataset': str(dataset_json),
        })
    (out_dir / 'metadata.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def generate(count: int, out_dir: str | Path, dataset_json: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    return generate_raven_questions(count=count, out_dir=Path(out_dir), dataset_json=dataset_json, seed=seed)


def generate(count: int, out_dir: str | Path, dataset_json: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    return generate_raven_questions(count=count, out_dir=Path(out_dir), dataset_json=dataset_json, seed=seed)
