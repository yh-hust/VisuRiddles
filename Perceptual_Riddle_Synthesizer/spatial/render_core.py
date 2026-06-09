from __future__ import annotations

import random
from typing import Iterable

from PIL import ImageDraw, ImageFont


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_connected_voxel(rng: random.Random, min_blocks: int = 5, max_blocks: int = 8) -> list[list[list[int]]]:
    target = rng.randint(min_blocks, max_blocks)
    occupied = {(1, 1, 0)}
    while len(occupied) < target:
        base = rng.choice(tuple(occupied))
        axis = rng.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)])
        cand = (base[0] + axis[0], base[1] + axis[1], base[2] + axis[2])
        if 0 <= cand[0] < 3 and 0 <= cand[1] < 3 and 0 <= cand[2] < 3:
            occupied.add(cand)
    matrix = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for x, y, z in occupied:
        matrix[z][y][x] = 1
    return matrix


def occupied_cells(voxel: list[list[list[int]]]) -> set[tuple[int, int, int]]:
    return {(x, y, z) for z in range(3) for y in range(3) for x in range(3) if voxel[z][y][x]}


def from_occupied(occ: Iterable[tuple[int, int, int]]) -> list[list[list[int]]]:
    matrix = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for x, y, z in occ:
        matrix[z][y][x] = 1
    return matrix


def project_voxel(voxel: list[list[list[int]]]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    top = [[1 if any(voxel[z][r][c] for z in range(3)) else 0 for c in range(3)] for r in range(3)]
    front = [[1 if any(voxel[z][r][c] for r in range(3)) else 0 for c in range(3)] for z in range(2, -1, -1)]
    side = [[1 if any(voxel[z][r][c] for c in range(3)) else 0 for r in range(3)] for z in range(2, -1, -1)]
    return top, front, side


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*matrix)]


def rotate_z(voxel: list[list[list[int]]]) -> list[list[list[int]]]:
    out = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for z in range(3):
        for y in range(3):
            for x in range(3):
                out[z][x][2 - y] = voxel[z][y][x]
    return out


def voxel_variants(voxel: list[list[list[int]]], needed: int, rng: random.Random) -> list[list[list[int]]]:
    variants = [voxel]
    seen = {_signature(voxel)}
    base = occupied_cells(voxel)
    while len(variants) < needed:
        occ = set(base)
        action = rng.choice(["remove_add", "toggle"])
        if action == "remove_add" and len(occ) > 3:
            occ.remove(rng.choice(tuple(occ)))
            for _ in range(20):
                x, y, z = rng.randrange(3), rng.randrange(3), rng.randrange(3)
                if (x, y, z) not in occ:
                    occ.add((x, y, z))
                    break
        else:
            x, y, z = rng.randrange(3), rng.randrange(3), rng.randrange(3)
            if (x, y, z) in occ and len(occ) > 4:
                occ.remove((x, y, z))
            else:
                occ.add((x, y, z))
        candidate = from_occupied(occ)
        sig = _signature(candidate)
        if sig not in seen and project_voxel(candidate) != project_voxel(voxel):
            variants.append(candidate)
            seen.add(sig)
    return variants


def matrix_variants(matrix: list[list[int]], needed: int, rng: random.Random) -> list[list[list[int]]]:
    variants = [matrix]
    seen = {_matrix_sig(matrix)}
    while len(variants) < needed:
        candidate = [row[:] for row in matrix]
        flips = 1 + rng.randrange(2)
        for _ in range(flips):
            r, c = rng.randrange(3), rng.randrange(3)
            candidate[r][c] = 1 - candidate[r][c]
        if sum(map(sum, candidate)) < 2:
            candidate[1][1] = 1
            candidate[0][1] = 1
        sig = _matrix_sig(candidate)
        if sig not in seen:
            variants.append(candidate)
            seen.add(sig)
    return variants


def draw_matrix(draw: ImageDraw.ImageDraw, matrix: list[list[int]], origin: tuple[int, int], cell: int, line_width: int = 2) -> None:
    ox, oy = origin
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            x0 = ox + c * cell
            y0 = oy + r * cell
            fill = "#3c3c3c" if value else "white"
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=fill)
    width = len(matrix[0]) * cell
    height = len(matrix) * cell
    for i in range(len(matrix) + 1):
        y = oy + i * cell
        draw.line((ox, y, ox + width, y), fill="black", width=line_width)
    for j in range(len(matrix[0]) + 1):
        x = ox + j * cell
        draw.line((x, oy, x, oy + height), fill="black", width=line_width)


def draw_cube(draw: ImageDraw.ImageDraw, x: int, y: int, s: int) -> None:
    top = [(x, y), (x + s, y - s // 2), (x + 2 * s, y), (x + s, y + s // 2)]
    left = [(x, y), (x + s, y + s // 2), (x + s, y + 3 * s // 2), (x, y + s)]
    right = [(x + 2 * s, y), (x + s, y + s // 2), (x + s, y + 3 * s // 2), (x + 2 * s, y + s)]
    draw.polygon(left, fill="#e2e2e2", outline="black")
    draw.polygon(right, fill="#cfcfcf", outline="black")
    draw.polygon(top, fill="white", outline="black")


def draw_isometric_voxel(draw: ImageDraw.ImageDraw, voxel: list[list[list[int]]], origin: tuple[int, int], cell: int) -> None:
    ox, oy = origin
    occ = sorted(occupied_cells(voxel), key=lambda xyz: (xyz[2], xyz[0] + xyz[1], xyz[0]))
    for x, y, z in occ:
        px = ox + (x - y) * cell
        py = oy + (x + y) * cell // 2 - z * cell
        draw_cube(draw, px, py, cell)


def draw_question_box(draw: ImageDraw.ImageDraw, origin: tuple[int, int], size: int) -> None:
    x, y = origin
    draw.rectangle([x, y, x + size, y + size], outline="black", width=3)
    draw.text((x + size // 2 - 10, y + size // 2 - 20), "?", fill="black", font=font(44))


def draw_net(draw: ImageDraw.ImageDraw, cells: list[tuple[int, int]], origin: tuple[int, int], cell: int, highlight: set[tuple[int, int]] | None = None) -> None:
    x_min = min(x for x, _ in cells)
    y_min = min(y for _, y in cells)
    for cx, cy in cells:
        x = origin[0] + (cx - x_min) * cell
        y = origin[1] + (cy - y_min) * cell
        fill = "#e8e8e8" if highlight and (cx, cy) in highlight else "white"
        draw.rectangle([x, y, x + cell, y + cell], fill=fill, outline="black", width=2)


def standard_nets() -> list[list[tuple[int, int]]]:
    return [
        [(1,0),(0,1),(1,1),(2,1),(1,2),(1,3)],
        [(0,0),(1,0),(2,0),(1,1),(1,2),(2,2)],
        [(0,1),(1,1),(2,1),(2,0),(2,2),(3,2)],
        [(1,0),(1,1),(0,1),(2,1),(2,2),(2,3)],
    ]


def _signature(voxel: list[list[list[int]]]) -> str:
    return ''.join(str(voxel[z][y][x]) for z in range(3) for y in range(3) for x in range(3))


def _matrix_sig(matrix: list[list[int]]) -> str:
    return ''.join(str(v) for row in matrix for v in row)
