from pathlib import Path

from synth_engine.config import load_config
from synth_engine.orchestrator import EngineOrchestrator


def test_spatial_positive_count_generates_questions(tmp_path):
    cfg = load_config(None)
    cfg["output_root"] = str(tmp_path / "out")
    for name in list(cfg["modules"].keys()):
        cfg["modules"][name]["enabled"] = False
    cfg["modules"]["spatial"]["enabled"] = True
    cfg["modules"]["spatial"]["count"] = 3
    manifest = EngineOrchestrator(cfg).run(["spatial"])
    spatial_record = manifest["modules"][0]
    assert spatial_record["status"] == "success"
    assert spatial_record["count_generated"] >= 3
    spatial_dir = Path(cfg["output_root"]) / "spatial"
    assert (spatial_dir / "index.json").exists()
    assert len(list(spatial_dir.glob("**/question.png"))) >= 3


def test_spatial_subcategories_have_distinct_visual_outputs(tmp_path):
    cfg = load_config(None)
    cfg["output_root"] = str(tmp_path / "out")
    for name in list(cfg["modules"].keys()):
        cfg["modules"][name]["enabled"] = False
    cfg["modules"]["spatial"]["enabled"] = True
    cfg["modules"]["spatial"]["count"] = 5
    cfg["modules"]["spatial"]["subrule_counts"] = {
        "unfolding": 1,
        "three_view": 1,
        "reconstruction_3d": 1,
        "view_consistency": 1,
        "multiple_views": 1,
    }
    EngineOrchestrator(cfg).run(["spatial"])
    images = [
        Path(cfg["output_root"]) / "spatial" / name / "question1" / "question.png"
        for name in cfg["modules"]["spatial"]["subrule_counts"]
    ]
    assert all(path.exists() for path in images)
    signatures = {path.read_bytes() for path in images}
    assert len(signatures) == len(images)
