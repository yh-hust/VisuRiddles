from __future__ import annotations

import importlib
from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.category_schema import CATEGORY_SCHEMA, split_count


class AttributeRunner(BaseRunner):
    name = "attribute"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        out_dir = output_root / module_cfg["output_subdir"]
        subrules = CATEGORY_SCHEMA["attribute"]
        if "subrule_counts" in module_cfg:
            counts = {name: int(module_cfg["subrule_counts"].get(name, 0)) for name in subrules}
            if sum(max(0, value) for value in counts.values()) == 0 and int(module_cfg.get("count", 0)) > 0:
                counts = split_count(int(module_cfg.get("count", 0)), subrules)
        else:
            counts = split_count(int(module_cfg.get("count", 0)), subrules)
            if "type2num" in module_cfg:
                total = sum(int(v) for v in module_cfg.get("type2num", {}).values())
                counts = split_count(total, subrules)
        requested = sum(max(0, v) for v in counts.values())
        if requested <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="attribute")
        generated = 0
        for subrule, count in counts.items():
            if count <= 0:
                continue
            mod = importlib.import_module(f"attribute.{subrule}")
            generated += int(mod.generate(count, out_dir, assets=module_cfg.get("assets", {})))
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=requested,
            count_generated=len(list(out_dir.glob("*.json"))) or generated,
            engine="attribute",
        )
