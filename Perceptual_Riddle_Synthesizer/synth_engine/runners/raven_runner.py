from __future__ import annotations

from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.utils import ensure_dir


class RavenRunner(BaseRunner):
    name = "raven"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        from raven.raven import generate

        out_dir = ensure_dir(output_root / module_cfg["output_subdir"])
        count = int(module_cfg.get("count", 0))
        if count <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="raven")
        records = generate(
            count=count,
            out_dir=out_dir,
            dataset_json=module_cfg.get("dataset_json"),
            seed=module_cfg.get("seed"),
        )
        metadata = out_dir / "metadata.json"
        warnings = []
        if module_cfg.get("dataset_json") is None:
            warnings.append("RAVEN generation uses the bundled symbolic dataset template.")
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=count,
            count_generated=len(records),
            engine="raven",
            artifacts=[str(metadata)] if metadata.exists() else [],
            warnings=warnings,
        )
