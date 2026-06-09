from __future__ import annotations

CATEGORY_SCHEMA = {
    "attribute": ["element", "group"],
    "positional": ["translate", "rotate", "flip"],
    "spatial": ["unfolding", "three_view", "reconstruction_3d", "view_consistency", "multiple_views"],
    "numerical": ["line", "curve", "angle", "cart", "space", "parts"],
    "stylistic": ["and", "or", "xor", "xnor"],
    "raven": ["raven"],
    "sudoku": ["sudoku"],
}

CATEGORY_ORDER = ["attribute", "positional", "spatial", "numerical", "stylistic", "raven", "sudoku"]


POSITIONAL_FINE_SUBRULES = {
    "translate": ["move_shift"],
    "rotate": ["self_rotate", "region_rotation", "rotation_grid", "icon_count_rotation", "transform_3x3"],
    "flip": ["mirror_flip", "flip_rotate_chain"],
}

PAPER_CATEGORY_ALIASES = {
    "Positional": "positional",
    "Spatial": "spatial",
    "Numerical": "numerical",
    "Stylistic": "stylistic",
    "Attribute": "attribute",
    "RAVEN": "raven",
    "Sudoku": "sudoku",
}


def split_count(total: int, names: list[str]) -> dict[str, int]:
    if not names:
        return {}
    total = max(0, int(total))
    base, remainder = divmod(total, len(names))
    return {name: base + (1 if idx < remainder else 0) for idx, name in enumerate(names)}
