import json
from pathlib import Path

from synth_engine.script_runner import build_category_config, run_category


def test_positional_fine_subrule_counts_are_supported(tmp_path):
    cfg = build_category_config(
        "positional",
        {"translate": 0, "rotate": 0, "flip": 0},
        str(tmp_path / "out"),
        fine_subrule_counts={"flip_rotate_chain": 2, "mirror_flip": 1, "self_rotate": 1},
    )
    loc_cfg = cfg["modules"]["positional"]
    assert loc_cfg["count"] == 4
    assert loc_cfg["subrule_counts"]["flip"] == 3
    assert loc_cfg["subrule_counts"]["rotate"] == 1
    assert loc_cfg["fine_grained_subrule_counts"]["flip_rotate_chain"] == 2


def test_positional_output_preserves_fine_grained_subrule_in_metadata_and_index(tmp_path):
    manifest = run_category(
        "positional",
        {"translate": 0, "rotate": 0, "flip": 0},
        str(tmp_path / "out"),
        fine_subrule_counts={"flip_rotate_chain": 2, "mirror_flip": 1},
    )
    assert manifest["raw_output_removed"] is True
    category_root = Path(tmp_path / "out" / "positional")
    assert (category_root / "flip" / "question1" / "metadata.json").exists()
    metadata = json.loads((category_root / "flip" / "question1" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rule"] == "flip"
    assert metadata["subrule"] in {"flip_rotate_chain", "mirror_flip"}
    category_index = json.loads((category_root / "index.json").read_text(encoding="utf-8"))
    assert category_index["count"] == 3
    subrules = {q.get("subrule") for q in category_index["questions"]}
    assert "flip_rotate_chain" in subrules
    assert "mirror_flip" in subrules
