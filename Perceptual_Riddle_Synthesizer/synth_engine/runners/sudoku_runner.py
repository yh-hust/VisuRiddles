from __future__ import annotations

from pathlib import Path

from .base import BaseRunner, RunResult
from synth_engine.utils import ensure_dir


class SudokuRunner(BaseRunner):
    name = "sudoku"

    def run(self, module_cfg: dict, output_root: Path) -> RunResult:
        from sudoku.sudoku import generate

        out_dir = ensure_dir(output_root / module_cfg["output_subdir"])
        count = int(module_cfg.get("count", 0))
        if count <= 0:
            return RunResult(module=self.name, status="skipped", output_dir=str(out_dir), count_requested=0, engine="sudoku")
        records = generate(
            count=count,
            out_dir=out_dir,
            levels=module_cfg.get("levels"),
            seed=module_cfg.get("seed"),
        )
        metadata = out_dir / "metadata.json"
        return RunResult(
            module=self.name,
            status="success",
            output_dir=str(out_dir),
            count_requested=count,
            count_generated=len(records),
            engine="sudoku",
            artifacts=[str(metadata)] if metadata.exists() else [],
        )
