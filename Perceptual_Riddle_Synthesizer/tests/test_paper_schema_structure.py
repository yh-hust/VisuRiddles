from pathlib import Path


def test_removed_legacy_generator_directories_and_imports():
    root = Path(__file__).resolve().parents[1]
    removed_names = [
        ("numerical", "num" + "_gen"),
        ("numerical", "num" + "_gen2"),
        ("positional", "generator"),
        ("positional", "run" + "_all.py"),
        ("attribute", "gen" + "_symmetry.py"),
        ("stylistic", "gen.py"),
        ("spatial", "generate" + "_spatial" + "_puzzles.py"),
        ("spatial", "three" + "_view" + "_voxel.py"),
        ("spatial", "voxel" + "_generator.py"),
    ]
    for parts in removed_names:
        path = root.joinpath(*parts)
        assert not path.exists(), f"Removed generator path should not exist: {path.relative_to(root)}"

    removed_terms = [
        "num" + "_gen",
        "num" + "_gen2",
        "gen" + "_symmetry",
        "positional" + "." + "generator",
        "run" + "_all",
        "generate" + "_spatial" + "_puzzles",
    ]
    for py_file in root.rglob("*.py"):
        if ".pytest_cache" in py_file.parts or py_file.name == "test_paper_schema_structure.py":
            continue
        text = py_file.read_text(encoding="utf-8")
        for term in removed_terms:
            assert term not in text, f"Removed reference {term!r} found in {py_file.relative_to(root)}"


def test_paper_aligned_subrule_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        "attribute/element.py",
        "attribute/group.py",
        "positional/translate.py",
        "positional/rotate.py",
        "positional/flip.py",
        "spatial/unfolding.py",
        "spatial/three_view.py",
        "spatial/reconstruction_3d.py",
        "spatial/view_consistency.py",
        "spatial/multiple_views.py",
        "numerical/line.py",
        "numerical/curve.py",
        "numerical/angle.py",
        "numerical/cart.py",
        "numerical/space.py",
        "numerical/parts.py",
        "stylistic/and.py",
        "stylistic/or.py",
        "stylistic/xor.py",
        "stylistic/xnor.py",
        "raven/raven.py",
        "sudoku/sudoku.py",
    ]
    for rel in expected:
        assert (root / rel).exists(), f"Missing paper-aligned generator file: {rel}"


def test_no_common_py_files_remain():
    root = Path(__file__).resolve().parents[1]
    assert not list(root.rglob("common.py"))


def test_removed_redundant_legacy_files():
    root = Path(__file__).resolve().parents[1]
    redundant_paths = [
        "raven/engine.py",
        "raven/shitu.py",
        "raven/sudoku.py",
        "raven/run.sh",
        "raven/setup_envs.sh",
        "sudoku/engine.py",
        "sudoku/main.py",
        "sudoku/show_GUI.py",
        "sudoku/show_function.py",
        "sudoku/sudoku_solver.py",
        "synth_engine/numerical_postprocess.py",
    ]
    for rel in redundant_paths:
        assert not (root / rel).exists(), f"Redundant legacy file should not exist: {rel}"

    for forbidden in ["raven.engine", "sudoku.engine", "PyQt5", "show_GUI", "sudoku_solver"]:
        for py_file in root.rglob("*.py"):
            if ".pytest_cache" in py_file.parts or py_file.name == "test_paper_schema_structure.py":
                continue
            assert forbidden not in py_file.read_text(encoding="utf-8")
