from __future__ import annotations

import importlib
from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.category_schema import CATEGORY_SCHEMA, POSITIONAL_FINE_SUBRULES, split_count


class PositionalRunner(BaseRunner):
    name = "positional"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        out_dir = output_root / module_cfg["output_subdir"]
        subrules = CATEGORY_SCHEMA["positional"]
        if "subrule_counts" in module_cfg:
            counts = {name: int(module_cfg["subrule_counts"].get(name, 0)) for name in subrules}
            if sum(max(0, value) for value in counts.values()) == 0 and int(module_cfg.get("count", 0)) > 0:
                counts = split_count(int(module_cfg.get("count", 0)), subrules)
        else:
            counts = split_count(int(module_cfg.get("count", 0)), subrules)
        requested = sum(max(0, v) for v in counts.values())
        if requested <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="positional")
        generated = 0
        fine_counts = module_cfg.get("fine_grained_subrule_counts", {}) or {}
        for subrule, count in counts.items():
            if count <= 0:
                continue
            mod = importlib.import_module(f"positional.{subrule}")
            scoped_fine_counts = {name: int(fine_counts.get(name, 0)) for name in POSITIONAL_FINE_SUBRULES.get(subrule, [])}
            generated += int(mod.generate(count, out_dir, resources=module_cfg.get("resources", {}), subrule_counts=scoped_fine_counts))
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=requested,
            count_generated=generated,
            engine="positional",
        )
