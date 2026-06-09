from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    module: str
    status: str
    output_dir: str
    count_requested: int = 0
    count_generated: int | None = None
    engine: str | None = None
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseRunner:
    name: str = "base"

    def run(self, module_cfg: dict[str, Any], output_root: Path) -> RunResult:
        raise NotImplementedError
