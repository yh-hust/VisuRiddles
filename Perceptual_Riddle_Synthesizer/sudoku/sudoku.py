from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .sudoku_creator import creator

DIGITS = set(range(1, 10))


def _normalize_board(board: Iterable[Iterable[object]]) -> list[list[int]]:
    out: list[list[int]] = []
    for row in board:
        new_row: list[int] = []
        for cell in row:
            if isinstance(cell, str):
                cell = cell.strip()
                new_row.append(int(cell) if cell and cell != '#' else 0)
            else:
                new_row.append(int(cell or 0))
        out.append(new_row)
    return out


def _find_empty(board: list[list[int]]) -> tuple[int, int] | None:
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return r, c
    return None


def _is_valid(board: list[list[int]], row: int, col: int, val: int) -> bool:
    if any(board[row][x] == val for x in range(9)):
        return False
    if any(board[y][col] == val for y in range(9)):
        return False
    br = (row // 3) * 3
    bc = (col // 3) * 3
    for y in range(br, br + 3):
        for x in range(bc, bc + 3):
            if board[y][x] == val:
                return False
    return True


def solve_sudoku_board(board: list[list[int]]) -> list[list[int]] | None:
    board = [row[:] for row in board]

    def backtrack() -> bool:
        pos = _find_empty(board)
        if pos is None:
            return True
        r, c = pos
        candidates = list(DIGITS - set(board[r]) - {board[y][c] for y in range(9)})
        random.shuffle(candidates)
        for val in candidates:
            if _is_valid(board, r, c, val):
                board[r][c] = val
                if backtrack():
                    return True
                board[r][c] = 0
        return False

    return board if backtrack() else None


def board_to_string(board: list[list[int]], blank: str = '#') -> str:
    lines = []
    for row in board:
        lines.append(''.join(str(v) if v else blank for v in row))
    return '\n'.join(lines)


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except Exception:
        return ImageFont.load_default()


def render_sudoku_board(board: list[list[int]], save_path: str | Path, title: str | None = None) -> None:
    cell = 72
    margin = 28
    title_h = 48 if title else 0
    size = margin * 2 + cell * 9
    img = Image.new('RGB', (size, size + title_h), 'white')
    draw = ImageDraw.Draw(img)

    light_a = (220, 255, 255)
    light_b = (225, 255, 235)
    for r in range(9):
        for c in range(9):
            x0 = margin + c * cell
            y0 = margin + title_h + r * cell
            x1 = x0 + cell
            y1 = y0 + cell
            use_b = (((0 <= r <= 2) or (6 <= r <= 8)) and (3 <= c <= 5)) or ((3 <= r <= 5) and ((0 <= c <= 2) or (6 <= c <= 8)))
            draw.rectangle([x0, y0, x1, y1], fill=light_b if use_b else light_a)

    for i in range(10):
        w = 4 if i % 3 == 0 else 1
        x = margin + i * cell
        y = margin + title_h + i * cell
        draw.line([(margin, title_h + margin + i * cell), (margin + 9 * cell, title_h + margin + i * cell)], fill='black', width=w)
        draw.line([(x, title_h + margin), (x, title_h + margin + 9 * cell)], fill='black', width=w)

    font = _get_font(30)
    title_font = _get_font(24)
    if title:
        draw.text((margin, 8), title, fill='black', font=title_font)
    for r in range(9):
        for c in range(9):
            v = board[r][c]
            if not v:
                continue
            text = str(v)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = margin + c * cell + (cell - tw) / 2
            y = title_h + margin + r * cell + (cell - th) / 2 - 2
            draw.text((x, y), text, fill='black', font=font)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def generate_sudoku_questions(count: int, out_dir: str | Path, levels: list[int] | None = None, seed: int | None = None) -> list[dict]:
    out_dir = Path(out_dir)
    questions_dir = out_dir / 'questions'
    answers_dir = out_dir / 'answers'
    questions_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    levels = levels or [1, 2, 3, 4, 5]

    gen = creator()
    records: list[dict] = []
    for idx in range(count):
        level = levels[idx % len(levels)]
        level = rng.choice(levels) if seed is not None else level
        puzzle = _normalize_board(gen.get_sudo_subject(level))
        solution = solve_sudoku_board(puzzle)
        if solution is None:
            raise RuntimeError(f'Failed to solve generated sudoku puzzle at index {idx}')
        q_path = questions_dir / f'q_{level}_{idx}.png'
        a_path = answers_dir / f'a_{level}_{idx}.png'
        render_sudoku_board(puzzle, q_path, title=f'Sudoku Level {level}')
        render_sudoku_board(solution, a_path, title=f'Sudoku Level {level} Answer')
        records.append({
            'id': idx,
            'level': level,
            'image_path': str(q_path),
            'answer_image_path': str(a_path),
            'question': board_to_string(puzzle),
            'answer': board_to_string(solution),
        })

    with (out_dir / 'metadata.json').open('w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def generate(count: int, out_dir: str | Path, levels: list[int] | None = None, seed: int | None = None) -> list[dict]:
    return generate_sudoku_questions(count=count, out_dir=Path(out_dir), levels=levels, seed=seed)
