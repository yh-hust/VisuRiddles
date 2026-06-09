from __future__ import annotations

import json
from pathlib import Path

from synth_engine.utils import ensure_dir
from .logic_base import generate_logic_tasks

SUBRULE = 'xor'
ENGINE_RULE = 'xor'
CANONICAL_RULE = 'xor'


def generate(count: int, out_dir: str | Path) -> int:
    total = max(0, int(count))
    if total <= 0:
        return 0
    subdir = ensure_dir(Path(out_dir) / SUBRULE)
    generate_logic_tasks(num=total, out_dir=str(subdir), rule_name=ENGINE_RULE)
    metadata_path = subdir / "metadata.json"
    records = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else []
    for record in records:
        if isinstance(record, dict):
            record["canonical_rule"] = CANONICAL_RULE
    metadata_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)
