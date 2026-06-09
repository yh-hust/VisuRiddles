from __future__ import annotations

import contextlib
import io
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from .config import load_config
from .format_normalizer import CANONICAL_CATEGORY_ORDER, normalize_module_output, write_global_index
from .registry import DEFAULT_ORDER, RUNNER_REGISTRY
from .utils import ensure_dir


class EngineOrchestrator:
    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def from_path(cls, config_path: str | None = None) -> "EngineOrchestrator":
        return cls(load_config(config_path))

    def run(self, selected_modules: Iterable[str] | None = None) -> dict:
        selected = list(selected_modules or DEFAULT_ORDER)
        output_root = ensure_dir(self.config["output_root"])
        raw_output_root = ensure_dir(self._raw_output_root(output_root))
        manifest = self._new_manifest(output_root, raw_output_root)

        normalization_cfg = dict(self.config.get("format_normalization", {}) or {})
        normalization_cfg["metadata_language"] = self.config.get("metadata_language", "zh")
        if normalization_cfg.get("enabled", True):
            self._prepare_final_output(output_root)

        try:
            self._run_modules(selected, output_root, raw_output_root, normalization_cfg, manifest)
            if normalization_cfg.get("enabled", True):
                manifest["dataset_index"] = str(write_global_index(output_root))
            self._write_manifest(raw_output_root, manifest)
            return manifest
        finally:
            self._cleanup_raw_output(raw_output_root, manifest)

    def _new_manifest(self, output_root: Path, raw_output_root: Path) -> dict:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "output_root": str(output_root),
            "raw_output_root": str(raw_output_root),
            "raw_output_removed": False,
            "continue_on_error": bool(self.config.get("continue_on_error", True)),
            "modules": [],
        }

    def _run_modules(
        self,
        selected: list[str],
        output_root: Path,
        raw_output_root: Path,
        normalization_cfg: dict,
        manifest: dict,
    ) -> None:
        progress = tqdm(selected, desc="Generating", unit="module", dynamic_ncols=True, ascii=True)
        for name in progress:
            self._run_one_module(name, output_root, raw_output_root, normalization_cfg, manifest)

    def _run_one_module(
        self,
        name: str,
        output_root: Path,
        raw_output_root: Path,
        normalization_cfg: dict,
        manifest: dict,
    ) -> None:
        runner = RUNNER_REGISTRY[name]
        module_cfg = self.config["modules"].get(name, {})
        raw_module_dir = raw_output_root / module_cfg.get("output_subdir", name)

        if not module_cfg.get("enabled", False):
            manifest["modules"].append({"module": name, "status": "disabled"})
            return

        try:
            with self._silence_child_output():
                result = runner.run(module_cfg, raw_output_root)
            module_record = result.to_dict()
            if module_record.get("status") == "success":
                norm_info = normalize_module_output(
                    name,
                    module_record.get("output_dir", raw_module_dir),
                    output_root,
                    normalization_cfg,
                )
                module_record["standardized"] = norm_info
                module_record.setdefault("artifacts", []).extend(norm_info.get("artifacts", []))
                if norm_info.get("warnings"):
                    module_record.setdefault("warnings", []).extend(norm_info["warnings"])
            manifest["modules"].append(module_record)
        except Exception as exc:
            manifest["modules"].append({"module": name, "status": "failed", "error": str(exc)})
            if not self.config.get("continue_on_error", True):
                self._write_manifest(raw_output_root, manifest)
                raise


    def _prepare_final_output(self, output_root: Path) -> None:
        index_path = output_root / "index.json"
        if index_path.exists():
            index_path.unlink()
        for category in CANONICAL_CATEGORY_ORDER:
            category_dir = output_root / category
            if category_dir.exists():
                shutil.rmtree(category_dir)
            ensure_dir(category_dir)

    def _raw_output_root(self, output_root: Path) -> Path:
        return output_root.parent / f"{output_root.name}_raw"

    @contextlib.contextmanager
    def _silence_child_output(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def _write_manifest(self, raw_output_root: Path, manifest: dict) -> None:
        (raw_output_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _cleanup_raw_output(self, raw_output_root: Path, manifest: dict) -> None:
        shutil.rmtree(raw_output_root, ignore_errors=True)
        manifest["raw_output_removed"] = True


def validate_output(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    result = {
        "output_root": str(output_root),
        "expected_categories": CANONICAL_CATEGORY_ORDER,
        "category_count": 0,
        "issues": [],
    }
    if not output_root.exists():
        result["issues"].append(f"output root is missing: {output_root}")
        return result
    category_dirs = [p.name for p in output_root.iterdir() if p.is_dir()]
    result["category_count"] = len(category_dirs)
    missing = [name for name in CANONICAL_CATEGORY_ORDER if not (output_root / name).is_dir()]
    extra = [name for name in category_dirs if name not in CANONICAL_CATEGORY_ORDER]
    if missing:
        result["issues"].append(f"missing category folders: {missing}")
    if extra:
        result["issues"].append(f"unexpected category folders: {extra}")
    for category in CANONICAL_CATEGORY_ORDER:
        cat_dir = output_root / category
        if cat_dir.exists() and not (cat_dir / "index.json").exists():
            has_question = any((p / "question.png").exists() for p in cat_dir.glob("**/question*"))
            if has_question:
                result["issues"].append(f"{category}: index.json is missing")
    return result
