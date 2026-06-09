from generate import build_parser
from synth_engine.generate_config import PUBLIC_CATEGORY_ARGS, build_generate_config


def _visible_options(parser):
    return {
        option
        for action in parser._actions
        if action.option_strings and action.help != "==SUPPRESS=="
        for option in action.option_strings
    }


def test_generate_py_first_public_arguments_are_canonical_categories():
    parser = build_parser()
    visible_optional_actions = [
        action for action in parser._actions
        if action.option_strings and action.help != "==SUPPRESS==" and "--help" not in action.option_strings
    ]
    first_seven = [action.option_strings[0].lstrip("-") for action in visible_optional_actions[:7]]
    assert first_seven == PUBLIC_CATEGORY_ARGS


def test_generate_py_category_counts_map_to_internal_modules(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "--attribute", "4",
        "--positional", "8",
        "--spatial", "3",
        "--numerical", "5",
        "--stylistic", "7",
        "--raven", "2",
        "--sudoku", "1",
        "--out_root", str(tmp_path / "out"),
    ])
    cfg = build_generate_config(args)

    assert sum(cfg["modules"]["attribute"]["subrule_counts"].values()) == 4
    assert cfg["modules"]["positional"]["count"] == 8
    assert sum(cfg["modules"]["positional"]["subrule_counts"].values()) == 8
    assert cfg["modules"]["spatial"]["count"] == 3
    assert sum(cfg["modules"]["spatial"]["subrule_counts"].values()) == 3
    assert sum(cfg["modules"]["numerical"]["subrule_counts"].values()) == 5
    assert sum(cfg["modules"]["stylistic"]["subrule_counts"].values()) == 7
    assert cfg["modules"]["raven"]["count"] == 2
    assert cfg["modules"]["sudoku"]["count"] == 1


def test_legacy_and_resource_path_arguments_are_not_public_or_supported():
    parser = build_parser()
    visible = _visible_options(parser)
    removed = [
        "--loc", "--voxel", "--style", "--style2", "--num", "--attr",
        "--style_lib", "--num_input", "--num_output", "--num_qimg",
        "--attr_font", "--attr_sym_root", "--attr_other_root", "--attr_json", "--attr_not_sym_root",
    ]
    for option in removed:
        assert option not in visible
    assert "--raw_output_root" not in visible
    for public_name in [f"--{name}" for name in PUBLIC_CATEGORY_ARGS]:
        assert public_name in visible


def test_raw_output_root_argument_is_removed(tmp_path):
    parser = build_parser()
    try:
        parser.parse_args(["--raw_output_root", str(tmp_path / "raw")])
    except SystemExit:
        return
    raise AssertionError("--raw_output_root should not be accepted")


def test_generate_py_does_not_define_config_builder():
    import generate

    assert not hasattr(generate, "build_compat_config")
    assert not hasattr(generate, "build_generate_config")
