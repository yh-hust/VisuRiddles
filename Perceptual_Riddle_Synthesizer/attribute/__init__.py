from __future__ import annotations

from pathlib import Path
from typing import Any

from synth_engine.category_schema import split_count

from . import element, group


def generate_attribute_questions(count: int, out_dir: str | Path, assets: dict[str, Any] | None = None) -> int:
    generated = 0
    subrules = {"element": element.generate, "group": group.generate}
    for name, sub_count in split_count(count, list(subrules)).items():
        generated += subrules[name](sub_count, out_dir, assets)
    return generated
