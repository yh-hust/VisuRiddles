from synth_engine.config import load_config
from synth_engine.registry import DEFAULT_ORDER, RUNNER_REGISTRY


def test_load_default_config():
    cfg = load_config(None)
    assert "modules" in cfg
    assert cfg["modules"]["positional"]["enabled"] is True


def test_registry_complete():
    for name in DEFAULT_ORDER:
        assert name in RUNNER_REGISTRY
