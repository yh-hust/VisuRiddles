from __future__ import annotations

from synth_engine.runners import (
    AttributeRunner,
    PositionalRunner,
    NumericalRunner,
    SpatialRunner,
    StylisticRunner,
    SudokuRunner,
    RavenRunner,
)

RUNNER_REGISTRY = {
    "attribute": AttributeRunner(),
    "positional": PositionalRunner(),
    "spatial": SpatialRunner(),
    "numerical": NumericalRunner(),
    "stylistic": StylisticRunner(),
    "raven": RavenRunner(),
    "sudoku": SudokuRunner(),
}

DEFAULT_ORDER = ["attribute", "positional", "spatial", "numerical", "stylistic", "raven", "sudoku"]
