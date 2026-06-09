import json
from pathlib import Path

from PIL import Image

from synth_engine.format_normalizer import normalize_module_output


def test_attribute_metadata_preserves_standalone_json_and_question_mark_source(tmp_path):
    raw = tmp_path / "raw_attribute"
    out = tmp_path / "out"
    raw.mkdir()
    meta = {
        "title": [
            "app-store-line.png",
            "disqus-line.png",
            "app-store-line-clone1.png",
            "meta-fill.png",
            "stackshare-line.png",
            "question_mark.png",
        ],
        "choices": ["gemini-line.png", "paypal-fill.png", "product-hunt-line.png", "java-fill.png"],
        "gt": "C",
        "type": "five2one_symmetry_type",
    }
    (raw / "sample.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    Image.new("RGB", (64, 64), "white").save(raw / "sample.png")

    normalize_module_output("attribute", raw, out, None)
    normalized = json.loads((out / "attribute" / "element" / "question1" / "metadata.json").read_text(encoding="utf-8"))

    assert normalized["details"] == meta
    assert normalized["answer"] == "C"
    assert "question_mark.png" in normalized["details"]["title"]
    assert "format_version" not in normalized
    assert "source" not in normalized
    assert "source_metadata" not in normalized
