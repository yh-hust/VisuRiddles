from __future__ import annotations

import importlib
from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.category_schema import CATEGORY_SCHEMA, split_count


class SpatialRunner(BaseRunner):
    name = "spatial"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        out_dir = output_root / module_cfg["output_subdir"]
        subrules = CATEGORY_SCHEMA["spatial"]
        if "subrule_counts" in module_cfg:
            counts = {name: int(module_cfg["subrule_counts"].get(name, 0)) for name in subrules}
            if sum(max(0, value) for value in counts.values()) == 0 and int(module_cfg.get("count", 0)) > 0:
                counts = split_count(int(module_cfg.get("count", 0)), subrules)
        else:
            counts = split_count(int(module_cfg.get("count", 0)), subrules)
        requested = sum(max(0, v) for v in counts.values())
        if requested <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="spatial")
        generated = 0
        for subrule, count in counts.items():
            if count <= 0:
                continue
            mod = importlib.import_module(f"spatial.{subrule}")
            generated += int(mod.generate(count, out_dir, seed=int(module_cfg.get("seed", 42))))
        summary = out_dir / "summary.json"
        if requested > 0 and generated == 0:
            raise RuntimeError("spatial generation requested a positive count but produced zero questions")
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=requested,
            count_generated=generated,
            engine="spatial",
            artifacts=[str(summary)] if summary.exists() else [],
        )
