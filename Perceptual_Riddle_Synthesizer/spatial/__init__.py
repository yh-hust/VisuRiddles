from __future__ import annotations

from pathlib import Path

from synth_engine.category_schema import split_count

from . import multiple_views, reconstruction_3d, three_view, unfolding, view_consistency

SUBRULE_GENERATORS = {
    "unfolding": unfolding.generate,
    "three_view": three_view.generate,
    "reconstruction_3d": reconstruction_3d.generate,
    "view_consistency": view_consistency.generate,
    "multiple_views": multiple_views.generate,
}


def generate_spatial_questions(count: int, out_dir: str | Path, resource_root: str | None = None, seed: int = 42) -> int:
    generated = 0
    for subrule, sub_count in split_count(count, list(SUBRULE_GENERATORS)).items():
        generated += SUBRULE_GENERATORS[subrule](sub_count, out_dir, seed=seed)
    return generated
