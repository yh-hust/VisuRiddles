from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Iterator


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def pushd(path: str | Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
