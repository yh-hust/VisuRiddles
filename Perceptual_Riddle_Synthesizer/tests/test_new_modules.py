import json
from pathlib import Path

from synth_engine.orchestrator import EngineOrchestrator
from synth_engine.config import load_config


def test_new_modules_in_default_order_and_config():
    cfg = load_config(None)
    assert 'sudoku' in cfg['modules']
    assert 'raven' in cfg['modules']


def test_sudoku_package_importable():
    import sudoku
    assert hasattr(sudoku, 'generate_sudoku_questions')


def test_raven_package_importable():
    import raven
    assert hasattr(raven, 'generate_raven_questions')


def test_minimal_run_for_new_modules(tmp_path):
    cfg = load_config(None)
    cfg['output_root'] = str(tmp_path / 'out')
    for name in list(cfg['modules'].keys()):
        cfg['modules'][name]['enabled'] = False
    cfg['modules']['sudoku']['enabled'] = True
    cfg['modules']['sudoku']['count'] = 2
    cfg['modules']['sudoku']['seed'] = 123
    cfg['modules']['raven']['enabled'] = True
    cfg['modules']['raven']['count'] = 2
    cfg['modules']['raven']['seed'] = 123
    manifest = EngineOrchestrator(cfg).run(['sudoku', 'raven'])
    mods = {m['module']: m for m in manifest['modules']}
    assert mods['sudoku']['status'] == 'success'
    assert mods['sudoku']['count_generated'] == 2
    assert manifest['raw_output_removed'] is True
    assert not Path(manifest['raw_output_root']).exists()
    assert Path(cfg['output_root'], 'sudoku', 'level_1', 'question1', 'question.png').exists()
    assert Path(cfg['output_root'], 'sudoku', 'level_1', 'question1', 'subimages', 'stem', 'main.png').exists()
    assert mods['raven']['status'] == 'success'
    assert mods['raven']['count_generated'] == 2
    assert Path(cfg['output_root'], 'raven', 'question1', 'question.png').exists()
    assert Path(cfg['output_root'], 'raven', 'question1', 'subimages', 'options', 'A.png').exists()
    root_index_path = Path(cfg['output_root'], 'index.json')
    assert root_index_path.exists()
    root_index = json.loads(root_index_path.read_text(encoding='utf-8'))
    assert 'format_version' not in root_index
    assert root_index['count'] == 4
    raven_index = json.loads(Path(cfg['output_root'], 'raven', 'index.json').read_text(encoding='utf-8'))
    assert 'format_version' not in raven_index
    assert 'modules' not in raven_index
    assert not list(Path(cfg['output_root']).glob('**/*_part*_index.json'))


def test_public_canonical_category_names():
    from synth_engine.format_normalizer import CANONICAL_CATEGORY_ORDER, _canonical_category_name

    assert CANONICAL_CATEGORY_ORDER == [
        "attribute", "positional", "spatial", "numerical", "stylistic", "raven", "sudoku"
    ]
    for category in CANONICAL_CATEGORY_ORDER:
        assert _canonical_category_name(category) == category
