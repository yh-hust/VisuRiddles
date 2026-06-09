from __future__ import annotations

import importlib
from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.category_schema import CATEGORY_SCHEMA, split_count
from synth_engine.utils import ensure_dir


class NumericalRunner(BaseRunner):
    name = "numerical"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        out_dir = ensure_dir(output_root / module_cfg["output_subdir"])
        subrules = CATEGORY_SCHEMA["numerical"]
        if "subrule_counts" in module_cfg:
            counts = {name: int(module_cfg["subrule_counts"].get(name, 0)) for name in subrules}
            if sum(max(0, value) for value in counts.values()) == 0 and int(module_cfg.get("count", 0)) > 0:
                counts = split_count(int(module_cfg.get("count", 0)), subrules)
        else:
            counts = split_count(int(module_cfg.get("count", 0)), subrules)
        requested = sum(max(0, v) for v in counts.values())
        if requested <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="numerical")
        generated = 0
        for subrule, count in counts.items():
            if count <= 0:
                continue
            mod = importlib.import_module(f"numerical.{subrule}")
            generated += int(mod.generate(count, out_dir, module_cfg["input_json"], module_cfg["qimage"]))
        artifacts = [str(path) for path in sorted(out_dir.glob("*/questions.json"))]
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=requested,
            count_generated=generated,
            engine="numerical",
            artifacts=artifacts,
        )
