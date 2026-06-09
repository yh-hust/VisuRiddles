from __future__ import annotations

import json
import math
import re
import shutil
import string
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except Exception:
    np = None

from PIL import Image, ImageChops, ImageDraw, ImageFont

warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes.*",
    category=UserWarning,
)

from .utils import ensure_dir, project_root


FORMAT_VERSION = "6.0"
QUESTION_IMAGE_NAME = "question.png"
COMPONENTS_DIRNAME = "subimages"
QUESTION_COMPONENT_DIRNAME = "stem"
OPTION_COMPONENT_DIRNAME = "options"
QUESTION_MARK_SENTINEL = "__QUESTION_MARK__"
BLANK_SENTINEL = "__BLANK__"
OPTION_LABEL_POOL = list(string.ascii_uppercase)
METADATA_CANDIDATES = ("question.json", "question1.json", "meta.json")
ATTRIBUTE_QUESTION_MARK_NAMES = {"question-mark.png", "question_mark.png", "questionmark.png"}

CANONICAL_CATEGORY_MAP = {
    "attribute": "attribute",
    "positional": "positional",
    "spatial": "spatial",
    "numerical": "numerical",
    "stylistic": "stylistic",
    "raven": "raven",
    "sudoku": "sudoku",
}
CANONICAL_CATEGORY_ORDER = ["attribute", "positional", "spatial", "numerical", "stylistic", "raven", "sudoku"]

PUBLIC_MODULE_INDEX_MAP = {category: category for category in CANONICAL_CATEGORY_ORDER}

POSITIONAL_RULE_TO_SUBRULE = {
    "move_shift": "translate",
    "self_rotate": "rotate",
    "region_rotation": "rotate",
    "rotation_grid": "rotate",
    "icon_count_rotation": "rotate",
    "transform_3x3": "rotate",
    "mirror_flip": "flip",
    "flip_rotate_chain": "flip",
}

ATTRIBUTE_RULE_TO_SUBRULE = {
    "five2one_symmetry_type": "element",
    "three2three_symmetry_type": "element",
    "four2one_symmetry_direction": "group",
    "four2one_symmetry_direct": "group",
    "three2three_symmetry_overall": "group",
}

RULE_NAME_MAP = {
    "求异": "find_difference",
    "求同": "find_same",
    "去同存异": "remove_same_keep_different",
    "去异存同": "remove_different_keep_same",
    "与": "and",
    "或": "or",
    "异或": "xor",
    "同或": "xnor",
    "恒定": "constant",
    "恒为偶数": "always_even",
    "恒为奇数": "always_odd",
    "先偶数后奇数": "even_odd_even_odd",
    "先奇数后偶数": "odd_even_odd_even",
    "even_odd_even_odd_even": "even_odd_even_odd",
    "odd_even_odd_even_odd": "odd_even_odd_even",
    "e-o-e-o": "even_odd_even_odd",
    "o-e-o-e": "odd_even_odd_even",
    "递增": "increasing",
    "递减": "decreasing",
    "黑白运算": "black_white_operation",
    "逻辑运算题": "logic_operation",
    "数量规律": "numerical_pattern",
}

CHINESE_CHAR_NAME_MAP = {
    "恒": "constant",
    "定": "fixed",
    "为": "is",
    "偶": "even",
    "奇": "odd",
    "数": "number",
    "递": "progression",
    "增": "increase",
    "减": "decrease",
    "求": "find",
    "异": "different",
    "同": "same",
    "去": "remove",
    "存": "keep",
    "与": "and",
    "或": "or",
    "黑": "black",
    "白": "white",
    "运": "operation",
    "算": "calculation",
    "题": "question",
    "双": "double",
    "星": "star",
    "号": "symbol",
    "小": "small",
    "圆": "circle",
    "点": "dot",
    "状": "pattern",
    "横": "horizontal",
    "竖": "vertical",
    "线": "line",
    "扇": "sector",
    "形": "shape",
    "区": "region",
}

METADATA_KEY_TRANSLATIONS = {
    "纹理": "texture",
}

METADATA_VALUE_TRANSLATIONS = {
    "黑白运算": "black_white_operation",
    "逻辑运算题": "logic_operation",
    "数量规律": "numerical_pattern",
    "求异": "find_difference",
    "求同": "find_same",
    "去同存异": "remove_same_keep_different",
    "去异存同": "remove_different_keep_same",
    "与": "and",
    "或": "or",
    "异或": "xor",
    "同或": "xnor",
    "恒定": "constant",
    "恒为偶数": "always_even",
    "恒为奇数": "always_odd",
    "先偶数后奇数": "even_odd_even_odd",
    "先奇数后偶数": "odd_even_odd_even",
    "六边形(10区)": "hexagon_10_regions",
    "六边形(12区)": "hexagon_12_regions",
    "4×4正方形": "4x4_square",
    "十字型": "cross_shape",
    "斜线": "diagonal_lines",
    "交叉网格": "cross_grid",
    "大圆点": "large_dots",
    "星号": "star_symbol",
    "小圆点": "small_dots",
    "横线": "horizontal_lines",
    "竖线": "vertical_lines",
    "点状": "dotted",
    "双斜线": "double_diagonal_lines",
    "扇形(4)": "sector_4_regions",
    "扇形(6)": "sector_6_regions",
    "扇形(8)": "sector_8_regions",
    "空": "empty",
    "小号": "small",
    "小中号": "small_medium",
    "中号": "medium",
    "大中号": "medium_large",
    "大号": "large",
    "黑色": "black",
    "深灰色": "dark_gray",
    "浅灰色": "light_gray",
    "亮白色": "bright_white",
    "白色": "white",
    "圆形": "circle",
    "三角形": "triangle",
    "四边形": "quadrilateral",
    "五边形": "pentagon",
    "六边形": "hexagon",
}

METADATA_PHRASE_TRANSLATIONS = {
    "依次是": "in order",
    "网格的内容": "grid content",
    "内容": "content",
    "元素": "elements",
    "网格": "grid",
    "旋转": "rotated",
    "度": "degrees",
    "的": " ",
}


def _translate_metadata_to_english(value: Any) -> Any:
    if isinstance(value, dict):
        return {_translate_metadata_key_to_english(str(k)): _translate_metadata_to_english(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_translate_metadata_to_english(v) for v in value]
    if isinstance(value, str):
        return _translate_metadata_string_to_english(value)
    return value


def _translate_metadata_key_to_english(key: str) -> str:
    if key in METADATA_KEY_TRANSLATIONS:
        return METADATA_KEY_TRANSLATIONS[key]
    return _translate_metadata_string_to_english(key) if _contains_chinese(key) else key


def _translate_metadata_string_to_english(text: str) -> str:
    if not _contains_chinese(text):
        return text
    if text in METADATA_VALUE_TRANSLATIONS:
        return METADATA_VALUE_TRANSLATIONS[text]
    translated = text
    translated = re.sub(r"第(\d+)个网格的内容：", r"Grid \1 content: ", translated)
    translated = re.sub(r"第(\d+)个网格的内容:", r"Grid \1 content: ", translated)
    translated = re.sub(r"有(\d+)个元素，依次是：", r"contains \1 elements in order: ", translated)
    translated = re.sub(r"有(\d+)个元素,依次是:", r"contains \1 elements in order: ", translated)
    translated = re.sub(r"旋转(-?\d+)度的", r"rotated \1 degrees ", translated)
    translated = re.sub(r"旋转(-?\d+)度", r"rotated \1 degrees", translated)
    translated = re.sub(r"六边形\((\d+)区\)", r"hexagon_\1_regions", translated)
    translated = re.sub(r"扇形\((\d+)\)", r"sector_\1_regions", translated)
    translated = re.sub(r"(\d+)×(\d+)正方形", r"\1x\2_square", translated)
    for zh, en in sorted(METADATA_VALUE_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(zh, en)
    for zh, en in sorted(METADATA_PHRASE_TRANSLATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        translated = translated.replace(zh, en)
    translated = re.sub(r"(small|medium|large|small_medium|medium_large)(dark_gray|light_gray|black|white|bright_white)", r"\1 \2", translated)
    translated = re.sub(r"(dark_gray|light_gray|black|white|bright_white)(circle|triangle|quadrilateral|pentagon|hexagon)", r"\1 \2", translated)
    translated = translated.replace("：", ": ").replace("，", ", ").replace("。", ". ")
    translated = translated.replace("、", ", ").replace("；", "; ").replace("（", "(").replace("）", ")")
    translated = re.sub(r"[\u4e00-\u9fff]+", _translate_unmapped_chinese_run, translated)
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def _translate_unmapped_chinese_run(match: re.Match[str]) -> str:
    parts = [CHINESE_CHAR_NAME_MAP.get(ch, f"u{ord(ch):x}") for ch in match.group(0)]
    return "_".join(parts)


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))



@dataclass(frozen=True)
class NormalizationSettings:
    enabled: bool = True
    output_subdir: str = ""
    cell_size: int = 160
    padding: int = 24
    gap: int = 18
    section_gap: int = 56
    border_width: int = 3
    label_height: int = 42
    font_path: str | None = None
    metadata_language: str = "zh"

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "NormalizationSettings":
        cfg = cfg or {}
        image_cfg = cfg.get("image", {}) or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            output_subdir=str(cfg.get("output_subdir", "")),
            cell_size=int(image_cfg.get("cell_size", 160)),
            padding=int(image_cfg.get("padding", 24)),
            gap=int(image_cfg.get("gap", 18)),
            section_gap=int(image_cfg.get("section_gap", 56)),
            border_width=int(image_cfg.get("border_width", 3)),
            label_height=int(image_cfg.get("label_height", 42)),
            font_path=cfg.get("font_path") or image_cfg.get("font_path"),
            metadata_language=str(cfg.get("metadata_language", "zh")).lower(),
        )


@dataclass
class NormalizedQuestion:
    id: str
    module: str
    rule: str | None
    output_dir: Path
    question_image: Path | None = None
    metadata_file: Path | None = None

    def to_manifest_record(self, normalized_root: Path) -> dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "rule": self.rule,
            "output_dir": _relpath(self.output_dir, normalized_root.parent),
            "question_image": _relpath(self.question_image, self.output_dir) if self.question_image else None,
            "metadata": _relpath(self.metadata_file, self.output_dir) if self.metadata_file else None,
            "components_dir": COMPONENTS_DIRNAME,
        }


def normalize_module_output(module_name: str, output_dir: str | Path, output_root: str | Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = NormalizationSettings.from_config(cfg)
    if not settings.enabled:
        return {"enabled": False, "count": 0, "output_dir": None, "artifacts": []}

    src = Path(output_dir)
    root = Path(output_root)
    category = canonical_category_name(module_name)
    normalized_root = ensure_dir(root / settings.output_subdir / category)

    if not src.exists():
        return {
            "enabled": True,
            "count": 0,
            "output_dir": str(normalized_root),
            "category": category,
            "artifacts": [],
            "warnings": [f"source output dir does not exist: {src}"],
        }

    _clear_previous_module_outputs(module_name, normalized_root)

    handlers = {
        "positional": _normalize_positional,
        "stylistic": _normalize_stylistic,
        "attribute": _normalize_attribute,
        "numerical": _normalize_number,
        "sudoku": _normalize_sudoku,
        "raven": _normalize_raven,
        "spatial": _normalize_spatial,
    }
    handler = handlers.get(module_name, _normalize_generic)
    questions, warnings = handler(module_name, src, normalized_root, settings)

    manifest = {
        "format_version": FORMAT_VERSION,
        "category": category,
        "module": module_name,
        "source_output_dir": str(src),
        "normalized_output_dir": str(normalized_root),
        "count": len(questions),
        "questions": [q.to_manifest_record(normalized_root) for q in questions],
        "warnings": warnings,
    }
    manifest_paths = _write_canonical_category_indexes(normalized_root, manifest)
    return {
        "enabled": True,
        "category": category,
        "count": len(questions),
        "output_dir": str(normalized_root),
        "artifacts": [str(path) for path in manifest_paths],
        "warnings": warnings,
    }




def _normalize_positional(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    for rule_dir in sorted((p for p in src.iterdir() if p.is_dir()), key=_natural_key):
        for q_dir in sorted((p for p in rule_dir.iterdir() if p.is_dir() and p.name.startswith("question")), key=_natural_key):
            meta_path = _first_existing(q_dir / name for name in METADATA_CANDIDATES)
            meta = _read_json(meta_path) if meta_path else {}
            source_question_files = _question_files_from_meta(q_dir, meta) or _numeric_images(q_dir)
            option_files = _option_images(q_dir)
            display_question_items = list(source_question_files[:-1]) + [QUESTION_MARK_SENTINEL] if source_question_files else []
            subrule = POSITIONAL_RULE_TO_SUBRULE.get(rule_dir.name, _safe_name(rule_dir.name))
            out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, subrule, q_dir.name))
            question_image = out_dir / QUESTION_IMAGE_NAME if display_question_items or option_files else None
            if question_image:
                _render_standard_multiple_choice(
                    display_question_items,
                    option_files,
                    question_image,
                    settings,
                    grid=_grid_from_positional_meta(meta, len(display_question_items)),
                )
            else:
                warnings.append(f"{q_dir}: no renderable question image found")
            component_info = _materialize_components(
                out_dir,
                settings,
                display_question_items,
                option_files,
                option_labels=_option_labels(len(option_files)),
            )
            metadata_file = _write_standard_metadata(
                out_dir,
                module=module,
                rule=subrule,
                qid=q_dir.name,
                answer=_extract_answer(meta),
                source_dir=q_dir,
                source_metadata=meta,
                source_metadata_file=meta_path,
                source_image=(q_dir / "final.png") if (q_dir / "final.png").exists() else None,
                source_question_files=source_question_files,
                source_option_files=option_files,
                component_info=component_info,
                metadata_language=settings.metadata_language,
            )
            records.append(NormalizedQuestion(q_dir.name, module, subrule, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_stylistic(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    metadata_paths = sorted(src.glob("*/metadata.json"), key=_natural_key)
    if (src / "metadata.json").exists():
        metadata_paths.insert(0, src / "metadata.json")
    for metadata_path in metadata_paths:
        sub_records, sub_warnings = _normalize_stylistic_records(module, metadata_path.parent, dst_root, settings)
        records.extend(sub_records)
        warnings.extend(sub_warnings)
    return records, warnings


def _normalize_stylistic_records(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    metadata_path = src / "metadata.json"
    items = _read_json(metadata_path) if metadata_path.exists() else []
    if not isinstance(items, list):
        items = []
    rule_counts: dict[str, int] = {}
    for idx, meta in enumerate(items, start=1):
        if not isinstance(meta, dict):
            continue
        qid = str(meta.get("id", idx))
        image_name = str(meta.get("image") or f"{idx}.png")
        qstem = Path(image_name).stem
        subimg_dir = src / "subimgs" / qstem
        question_files = [subimg_dir / f"{i}.png" for i in range(1, 9) if (subimg_dir / f"{i}.png").exists()]
        option_files = [subimg_dir / f"{label}.png" for label in OPTION_LABEL_POOL[:4] if (subimg_dir / f"{label}.png").exists()]
        display_question_items = list(question_files) + [QUESTION_MARK_SENTINEL]
        rule = str(meta.get("canonical_rule") or meta.get("rule") or meta.get("type") or "") or None
        rule = _canonical_rule_name(rule) if rule else None
        canonical_qid = _next_canonical_qid(rule_counts, rule)
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_image = out_dir / QUESTION_IMAGE_NAME if question_files and option_files else None
        if question_image:
            _render_standard_multiple_choice(
                display_question_items,
                option_files,
                question_image,
                settings,
                grid=_grid_from_style_grid(meta.get("grid"), len(display_question_items)),
            )
        else:
            image_path = _coerce_existing_path(image_name, src)
            if image_path and image_path.exists():
                question_image = out_dir / QUESTION_IMAGE_NAME
                _copy_image(image_path, question_image)
            else:
                warnings.append(f"stylistic record {qid}: no renderable assets found")
                question_image = None
        component_info = _materialize_components(
            out_dir,
            settings,
            display_question_items if question_files else [],
            option_files,
            option_labels=_option_labels(len(option_files)),
        )
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=rule,
            qid=qid,
            answer=meta.get("answer"),
            source_dir=src,
            source_metadata=meta,
            source_metadata_file=metadata_path,
            source_image=_coerce_existing_path(image_name, src),
            source_question_files=question_files + ([subimg_dir / "9.png"] if (subimg_dir / "9.png").exists() else []),
            source_option_files=option_files,
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(qid, module, rule, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_attribute(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    resolver = _attribute_icon_resolver()
    rule_counts: dict[str, int] = {}
    for seq, meta_path in enumerate(sorted(src.glob("*.json"), key=_natural_key), start=1):
        if meta_path.name == "index.json":
            continue
        meta = _read_json(meta_path)
        stem = meta_path.stem
        title = meta.get("title") if isinstance(meta, dict) else None
        choices = meta.get("choices") if isinstance(meta, dict) else None
        display_question_items, question_grid, source_question_files = _attribute_question_items(title, resolver)
        option_files = [resolver(name) for name in choices] if isinstance(choices, list) else []
        option_files = [p for p in option_files if p]
        raw_rule = str(meta.get("type") or meta.get("rule") or stem) if isinstance(meta, dict) else stem
        rule = ATTRIBUTE_RULE_TO_SUBRULE.get(raw_rule, _safe_name(raw_rule))
        canonical_qid = _next_canonical_qid(rule_counts, rule)
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_image = out_dir / QUESTION_IMAGE_NAME if display_question_items else None
        if question_image:
            _render_standard_multiple_choice(
                display_question_items,
                option_files,
                question_image,
                settings,
                grid=question_grid,
            )
        else:
            raw_img = _first_existing(src / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp"))
            if raw_img:
                question_image = out_dir / QUESTION_IMAGE_NAME
                _copy_image(raw_img, question_image)
            else:
                warnings.append(f"{meta_path}: matching image not found")
                question_image = None
        component_info = _materialize_components(
            out_dir,
            settings,
            display_question_items,
            option_files,
            option_labels=_option_labels(len(option_files)),
        )
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=rule,
            qid=stem,
            answer=_extract_answer(meta),
            source_dir=src,
            source_metadata=meta,
            source_metadata_file=meta_path,
            source_image=_first_existing(src / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp")),
            source_question_files=source_question_files,
            source_option_files=option_files,
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(stem, module, rule, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_number(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    questions_paths = []
    if (src / "questions.json").exists():
        questions_paths.append(src / "questions.json")
    questions_paths.extend(sorted(src.glob("*/questions.json"), key=_natural_key))
    rule_counts: dict[str, int] = {}
    for questions_path in questions_paths:
        questions = _read_json(questions_path) if questions_path.exists() else []
        if not isinstance(questions, list):
            continue
        base_dir = questions_path.parent
        default_rule = base_dir.name if base_dir != src else None
        for idx, meta in enumerate(questions, start=1):
            if not isinstance(meta, dict):
                continue
            qid = str(meta.get("id", idx))
            cells = meta.get("cells") if isinstance(meta.get("cells"), list) else []
            answer_cells = meta.get("answer_cells") if isinstance(meta.get("answer_cells"), list) else []
            question_files = []
            for cell in cells:
                if isinstance(cell, dict):
                    p = _coerce_existing_path(cell.get("cell"), base_dir)
                    if p:
                        question_files.append(p)
            if question_files and _looks_like_question_mark_path(question_files[-1]):
                hidden_or_q = question_files.pop()
                source_question_files = question_files + [hidden_or_q]
            else:
                source_question_files = list(question_files)
            display_question_items = list(question_files) + [QUESTION_MARK_SENTINEL] if question_files else []
            option_items: list[Path] = []
            option_labels = []
            for answer_cell in answer_cells:
                if not isinstance(answer_cell, dict):
                    continue
                for label in _option_labels(26):
                    if label in answer_cell:
                        p = _coerce_existing_path(answer_cell.get(label), base_dir)
                        if p:
                            option_labels.append(label)
                            option_items.append(p)
                        break
            rule = str(meta.get("canonical_rule") or default_rule or _normalize_numerical_rule(meta))
            rule = _canonical_rule_name(rule)
            canonical_qid = _next_canonical_qid(rule_counts, rule)
            out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
            question_image = out_dir / QUESTION_IMAGE_NAME if display_question_items and option_items else None
            if question_image:
                _render_standard_multiple_choice(
                    display_question_items,
                    option_items,
                    question_image,
                    settings,
                    grid=_grid_from_style_grid(meta.get("grid"), len(display_question_items)),
                    option_labels=option_labels or None,
                )
            else:
                img = base_dir / f"question_{qid}.png"
                if img.exists():
                    question_image = out_dir / QUESTION_IMAGE_NAME
                    _copy_image(img, question_image)
                else:
                    warnings.append(f"{base_dir}: numerical preview image not found for record {qid}")
                    question_image = None
            component_info = _materialize_components(
                out_dir,
                settings,
                display_question_items,
                option_items,
                option_labels=option_labels or _option_labels(len(option_items)),
            )
            metadata_file = _write_standard_metadata(
                out_dir,
                module=module,
                rule=rule,
                qid=qid,
                answer=meta.get("answer"),
                source_dir=base_dir,
                source_metadata=meta,
                source_metadata_file=questions_path,
                source_image=(base_dir / f"question_{qid}.png") if (base_dir / f"question_{qid}.png").exists() else None,
                source_question_files=source_question_files,
                source_option_files=option_items,
                component_info=component_info,
                metadata_language=settings.metadata_language,
            )
            records.append(NormalizedQuestion(qid, module, rule, out_dir, question_image, metadata_file))
    return records, warnings



def _normalize_sudoku(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    metadata_path = src / "metadata.json"
    records_meta = _read_json(metadata_path) if metadata_path.exists() else []
    if not isinstance(records_meta, list):
        records_meta = []
    rule_counts: dict[str, int] = {}
    for idx, meta in enumerate(records_meta):
        if not isinstance(meta, dict):
            continue
        qid = str(meta.get("id", idx))
        question_src = _coerce_existing_path(meta.get("image_path"), src)
        answer_src = _coerce_existing_path(meta.get("answer_image_path"), src)
        rule = f"level_{meta.get('level')}" if meta.get("level") is not None else None
        canonical_qid = _next_canonical_qid(rule_counts, rule)
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_image = out_dir / QUESTION_IMAGE_NAME if question_src and question_src.exists() else None
        if question_image:
            _copy_image(question_src, question_image)
        else:
            warnings.append(f"sudoku record {qid}: question image not found")
        component_info = _materialize_components(
            out_dir,
            settings,
            [],
            [],
            question_panel_item=question_src if question_src and question_src.exists() else None,
            option_labels=[],
        )
        if answer_src and answer_src.exists():
            _copy_image(answer_src, out_dir / "answer.png")
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=f"level_{meta.get('level')}" if meta.get("level") is not None else None,
            qid=qid,
            answer=meta.get("answer"),
            source_dir=src,
            source_metadata=meta,
            source_metadata_file=metadata_path,
            source_image=question_src,
            source_question_files=[question_src] if question_src else [],
            source_option_files=[],
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(qid, module, f"level_{meta.get('level')}" if meta.get("level") is not None else None, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_raven(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    metadata_path = src / "metadata.json"
    items = _read_json(metadata_path) if metadata_path.exists() else []
    if not isinstance(items, list):
        items = []
    rule_counts: dict[str, int] = {}
    for idx, meta in enumerate(items):
        if not isinstance(meta, dict):
            continue
        qid = str(meta.get("id", idx))
        content_img = _coerce_existing_path(meta.get("content_image_path"), src)
        choices_img = _coerce_existing_path(meta.get("choices_image_path"), src)
        rule = str(meta.get("rule") or "") or None
        canonical_qid = _next_canonical_qid(rule_counts, rule)
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_cells = _extract_raven_grid_cells(content_img, rows=3, cols=3, remove_last=True) if content_img and content_img.exists() else []
        option_cells = _extract_raven_grid_cells(choices_img, rows=2, cols=4, remove_last=False) if choices_img and choices_img.exists() else []
        display_question_items = list(question_cells) + [QUESTION_MARK_SENTINEL] if question_cells else []
        question_image = out_dir / QUESTION_IMAGE_NAME if display_question_items and option_cells else None
        if question_image:
            _render_standard_multiple_choice(
                display_question_items,
                option_cells,
                question_image,
                settings,
                grid=(3, 3),
                option_cols=4,
                option_labels=_option_labels(len(option_cells)),
            )
        else:
            whole = _coerce_existing_path(meta.get("image_path"), src)
            if whole and whole.exists():
                question_image = out_dir / QUESTION_IMAGE_NAME
                _copy_image(whole, question_image)
            else:
                warnings.append(f"raven record {qid}: renderable assets not found")
                question_image = None
        component_info = _materialize_components(
            out_dir,
            settings,
            display_question_items,
            option_cells,
            option_labels=_option_labels(len(option_cells)),
        )
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=str(meta.get("rule") or "") or None,
            qid=qid,
            answer=_raven_answer_label(meta.get("answer"), len(option_cells)),
            source_dir=src,
            source_metadata=meta,
            source_metadata_file=metadata_path,
            source_image=_coerce_existing_path(meta.get("image_path"), src),
            source_question_files=[content_img] if content_img else [],
            source_option_files=[choices_img] if choices_img else [],
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(qid, module, str(meta.get("rule") or "") or None, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_spatial(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    records: list[NormalizedQuestion] = []
    warnings: list[str] = []
    summary_path = src / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else []
    if not isinstance(summary, list):
        summary = []
    rule_counts: dict[str, int] = {}
    for idx, meta in enumerate(summary):
        if not isinstance(meta, dict):
            continue
        qid = str(meta.get("id", idx))
        rule = str(meta.get("rule", "")) or None
        canonical_qid = _next_canonical_qid(rule_counts, rule)
        source_dir = _find_spatial_question_dir(src, qid, rule)
        image_src = _find_spatial_render(source_dir, qid) if source_dir else None
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_panel = None
        option_cells: list[Image.Image] = []
        if image_src and image_src.exists():
            question_panel, option_cells = _extract_spatial_assets(image_src)
        question_image = out_dir / QUESTION_IMAGE_NAME if (question_panel is not None and option_cells) else None
        if question_image:
            _render_standard_panel_choices(
                question_panel,
                option_cells,
                question_image,
                settings,
                option_cols=4,
                option_labels=_option_labels(len(option_cells)),
            )
        elif image_src and image_src.exists():
            question_image = out_dir / QUESTION_IMAGE_NAME
            _copy_image(image_src, question_image)
            warnings.append(f"spatial record {qid}: fell back to original composite render")
        else:
            warnings.append(f"spatial record {qid}: rendered puzzle image not found")
            question_image = None
        component_info = _materialize_components(
            out_dir,
            settings,
            [],
            option_cells,
            question_panel_item=question_panel,
            option_labels=_option_labels(len(option_cells)),
        )
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=rule,
            qid=qid,
            answer=meta.get("correct_answer"),
            source_dir=source_dir or src,
            source_metadata=meta,
            source_metadata_file=summary_path,
            source_image=image_src,
            source_question_files=[image_src] if image_src else [],
            source_option_files=[],
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(qid, module, rule, out_dir, question_image, metadata_file))
    return records, warnings


def _normalize_generic(module: str, src: Path, dst_root: Path, settings: NormalizationSettings) -> tuple[list[NormalizedQuestion], list[str]]:
    warnings = [f"no module-specific normalizer registered for {module}; copied top-level png/json pairs only"]
    records: list[NormalizedQuestion] = []
    for seq, meta_path in enumerate(sorted(src.glob("*.json"), key=_natural_key), start=1):
        if meta_path.name == "index.json":
            continue
        stem = meta_path.stem
        img = _first_existing(src / f"{stem}{suffix}" for suffix in (".png", ".jpg", ".jpeg", ".webp"))
        meta = _read_json(meta_path)
        rule = str(meta.get("rule") or stem) if isinstance(meta, dict) else stem
        canonical_qid = f"question{seq}"
        out_dir = ensure_dir(dst_root / _canonical_question_dir_name(module, rule, canonical_qid))
        question_image = out_dir / QUESTION_IMAGE_NAME if img else None
        if img and question_image:
            _copy_image(img, question_image)
        component_info = _materialize_components(
            out_dir,
            settings,
            [],
            [],
            question_panel_item=img if img else None,
            option_labels=[],
        )
        metadata_file = _write_standard_metadata(
            out_dir,
            module=module,
            rule=rule,
            qid=stem,
            answer=_extract_answer(meta),
            source_dir=src,
            source_metadata=meta,
            source_metadata_file=meta_path,
            source_image=img,
            source_question_files=[img] if img else [],
            source_option_files=[],
            component_info=component_info,
            metadata_language=settings.metadata_language,
        )
        records.append(NormalizedQuestion(stem, module, str(meta.get("rule") or stem) if isinstance(meta, dict) else stem, out_dir, question_image, metadata_file))
    return records, warnings




def _render_standard_multiple_choice(
    question_items: list[Any],
    option_items: list[Any],
    output_path: Path,
    settings: NormalizationSettings,
    *,
    grid: tuple[int, int] | None = None,
    option_cols: int | None = None,
    option_labels: list[str] | None = None,
) -> None:
    grid = grid or _auto_grid(len(question_items))
    q_rows, q_cols = grid
    tile = settings.cell_size
    pad = settings.padding
    section_gap = settings.section_gap
    label_h = settings.label_height
    gap = settings.gap
    bw = settings.border_width
    q_font = _load_font(settings.font_path, max(64, int(tile * 0.62)))
    label_font = _load_font(settings.font_path, max(24, int(tile * 0.18)))

    q_needed = q_rows * q_cols
    q_display = list(question_items)
    if len(q_display) < q_needed:
        q_display.extend([BLANK_SENTINEL] * (q_needed - len(q_display)))

    n_options = len(option_items)
    option_labels = option_labels or _option_labels(n_options)
    option_cols = option_cols or min(4, max(1, n_options))
    option_rows = max(1, math.ceil(max(1, n_options) / option_cols))
    top_w = q_cols * tile
    top_h = q_rows * tile
    bottom_w = option_cols * tile
    bottom_h = option_rows * tile
    option_section_h = option_rows * (tile + label_h)
    canvas_w = max(top_w, bottom_w) + pad * 2
    canvas_h = pad + top_h + (section_gap if n_options else 0) + (option_section_h if n_options else 0) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    top_x = (canvas_w - top_w) // 2
    top_y = pad
    _draw_grid_box(draw, top_x, top_y, q_cols, q_rows, tile, bw)
    for idx in range(q_needed):
        r, c = divmod(idx, q_cols)
        x = top_x + c * tile
        y = top_y + r * tile
        _paste_item(canvas, draw, q_display[idx], x, y, tile, q_font)

    if n_options:
        opt_x = (canvas_w - bottom_w) // 2
        opt_y = top_y + top_h + section_gap
        for idx, item in enumerate(option_items):
            r, c = divmod(idx, option_cols)
            x = opt_x + c * tile
            y = opt_y + r * (tile + label_h)
            draw.rectangle([x, y, x + tile, y + tile], outline="black", width=bw)
            _paste_item(canvas, draw, item, x, y, tile, q_font)
            label = option_labels[idx] if idx < len(option_labels) else str(idx + 1)
            bbox = draw.textbbox((0, 0), label, font=label_font)
            lx = x + tile // 2 - (bbox[2] - bbox[0]) // 2
            ly = y + tile + 4
            draw.text((lx, ly), label, fill="black", font=label_font)
    ensure_dir(output_path.parent)
    canvas.save(output_path)


def _render_standard_panel_choices(
    question_panel_item: Any,
    option_items: list[Any],
    output_path: Path,
    settings: NormalizationSettings,
    *,
    option_cols: int | None = None,
    option_labels: list[str] | None = None,
) -> None:
    tile = settings.cell_size
    pad = settings.padding
    section_gap = settings.section_gap
    label_h = settings.label_height
    bw = settings.border_width
    q_font = _load_font(settings.font_path, max(64, int(tile * 0.62)))
    label_font = _load_font(settings.font_path, max(24, int(tile * 0.18)))
    n_options = len(option_items)
    option_cols = option_cols or min(4, max(1, n_options))
    option_rows = max(1, math.ceil(max(1, n_options) / option_cols))
    option_labels = option_labels or _option_labels(n_options)

    panel_img = _item_to_image(question_panel_item, tile=tile * 3)
    panel_max_w = tile * 4
    panel_max_h = tile * 4
    panel_img.thumbnail((panel_max_w, panel_max_h), Image.Resampling.LANCZOS)
    panel_box_w = max(panel_img.width + pad * 2, tile * 3)
    panel_box_h = max(panel_img.height + pad * 2, tile * 2)
    option_w = option_cols * tile
    option_h = option_rows * tile
    option_section_h = option_rows * (tile + label_h)
    canvas_w = max(panel_box_w, option_w) + pad * 2
    canvas_h = pad + panel_box_h + section_gap + option_section_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    panel_x = (canvas_w - panel_box_w) // 2
    panel_y = pad
    draw.rectangle([panel_x, panel_y, panel_x + panel_box_w, panel_y + panel_box_h], outline="black", width=bw)
    px = panel_x + (panel_box_w - panel_img.width) // 2
    py = panel_y + (panel_box_h - panel_img.height) // 2
    canvas.paste(panel_img.convert("RGB"), (px, py))

    opt_x = (canvas_w - option_w) // 2
    opt_y = panel_y + panel_box_h + section_gap
    for idx, item in enumerate(option_items):
        r, c = divmod(idx, option_cols)
        x = opt_x + c * tile
        y = opt_y + r * (tile + label_h)
        draw.rectangle([x, y, x + tile, y + tile], outline="black", width=bw)
        _paste_item(canvas, draw, item, x, y, tile, q_font)
        label = option_labels[idx] if idx < len(option_labels) else str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=label_font)
        lx = x + tile // 2 - (bbox[2] - bbox[0]) // 2
        ly = y + tile + 4
        draw.text((lx, ly), label, fill="black", font=label_font)

    ensure_dir(output_path.parent)
    canvas.save(output_path)


def _draw_grid_box(draw: ImageDraw.ImageDraw, x: int, y: int, cols: int, rows: int, tile: int, bw: int) -> None:
    draw.rectangle([x, y, x + cols * tile, y + rows * tile], outline="black", width=bw)
    for c in range(1, cols):
        xx = x + c * tile
        draw.line([xx, y, xx, y + rows * tile], fill="black", width=bw)
    for r in range(1, rows):
        yy = y + r * tile
        draw.line([x, yy, x + cols * tile, yy], fill="black", width=bw)


def _paste_item(canvas: Image.Image, draw: ImageDraw.ImageDraw, item: Any, x: int, y: int, tile: int, q_font: ImageFont.ImageFont) -> None:
    margin = max(8, tile // 12)
    if item == QUESTION_MARK_SENTINEL:
        text = "?"
        bbox = draw.textbbox((0, 0), text, font=q_font)
        tx = x + tile // 2 - (bbox[2] - bbox[0]) // 2
        ty = y + tile // 2 - (bbox[3] - bbox[1]) // 2 - bbox[1]
        draw.text((tx, ty), text, fill="black", font=q_font)
        return
    if item == BLANK_SENTINEL or item is None:
        return
    img = _item_to_image(item, tile=tile)
    img.thumbnail((tile - 2 * margin, tile - 2 * margin), Image.Resampling.LANCZOS)
    bg = Image.new("RGBA", (tile - 2 * margin, tile - 2 * margin), (255, 255, 255, 0))
    ox = (bg.width - img.width) // 2
    oy = (bg.height - img.height) // 2
    bg.alpha_composite(img.convert("RGBA"), (ox, oy))
    canvas.paste(bg.convert("RGB"), (x + margin, y + margin))


def _materialize_components(
    out_dir: Path,
    settings: NormalizationSettings,
    question_items: list[Any],
    option_items: list[Any],
    *,
    question_panel_item: Any | None = None,
    option_labels: list[str] | None = None,
) -> dict[str, Any]:
    components_root = ensure_dir(out_dir / COMPONENTS_DIRNAME)
    question_dir = ensure_dir(components_root / QUESTION_COMPONENT_DIRNAME)
    option_dir = ensure_dir(components_root / OPTION_COMPONENT_DIRNAME)

    question_saved: list[str] = []
    option_saved: list[str] = []

    if question_panel_item is not None:
        dst = question_dir / "main.png"
        _save_item(question_panel_item, dst, settings.cell_size * 3)
        question_saved.append(str(dst.relative_to(out_dir)))

    for idx, item in enumerate(question_items, start=1):
        dst = question_dir / f"{idx:02d}.png"
        _save_item(item, dst, settings.cell_size)
        question_saved.append(str(dst.relative_to(out_dir)))

    labels = option_labels or _option_labels(len(option_items))
    for idx, item in enumerate(option_items):
        label = labels[idx] if idx < len(labels) else f"option_{idx+1}"
        dst = option_dir / f"{label}.png"
        _save_item(item, dst, settings.cell_size)
        option_saved.append(str(dst.relative_to(out_dir)))

    return {
        "components_dir": COMPONENTS_DIRNAME,
        "question_dir": str((components_root / QUESTION_COMPONENT_DIRNAME).relative_to(out_dir)),
        "options_dir": str((components_root / OPTION_COMPONENT_DIRNAME).relative_to(out_dir)),
        "question_files": question_saved,
        "option_files": option_saved,
    }


def _save_item(item: Any, dst: Path, tile: int) -> None:
    ensure_dir(dst.parent)
    if item == QUESTION_MARK_SENTINEL:
        q_font = _load_font(None, max(64, int(tile * 0.62)))
        img = Image.new("RGB", (tile, tile), "white")
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), "?", font=q_font)
        tx = tile // 2 - (bbox[2] - bbox[0]) // 2
        ty = tile // 2 - (bbox[3] - bbox[1]) // 2 - bbox[1]
        draw.text((tx, ty), "?", fill="black", font=q_font)
        img.save(dst)
        return
    if item == BLANK_SENTINEL or item is None:
        Image.new("RGB", (tile, tile), "white").save(dst)
        return
    if isinstance(item, Image.Image):
        item.convert("RGB").save(dst)
        return
    _copy_image(Path(item), dst)


def _copy_image(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    with Image.open(src) as im:
        im.convert("RGB").save(dst)


def _item_to_image(item: Any, tile: int) -> Image.Image:
    if item == QUESTION_MARK_SENTINEL:
        q_font = _load_font(None, max(64, int(tile * 0.62)))
        img = Image.new("RGBA", (tile, tile), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), "?", font=q_font)
        tx = tile // 2 - (bbox[2] - bbox[0]) // 2
        ty = tile // 2 - (bbox[3] - bbox[1]) // 2 - bbox[1]
        draw.text((tx, ty), "?", fill="black", font=q_font)
        return img
    if item == BLANK_SENTINEL or item is None:
        return Image.new("RGBA", (tile, tile), (255, 255, 255, 0))
    if isinstance(item, Image.Image):
        return item.convert("RGBA")
    with Image.open(Path(item)) as im:
        return im.convert("RGBA")


def _write_standard_metadata(
    out_dir: Path,
    *,
    module: str,
    rule: str | None,
    qid: str,
    answer: Any,
    source_dir: Path,
    source_metadata: Any,
    source_metadata_file: Path | None,
    source_image: Path | None,
    source_question_files: list[Path],
    source_option_files: list[Path],
    component_info: dict[str, Any],
    metadata_language: str = "zh",
) -> Path:
    canonical_rule = _safe_name(rule) if rule else None
    data: dict[str, Any] = {
        "id": out_dir.name,
        "category": canonical_category_name(module),
        "rule": canonical_rule,
        "answer": answer,
        "question_image": QUESTION_IMAGE_NAME,
        "stem_images": component_info.get("question_files", []),
        "option_images": component_info.get("option_files", []),
    }
    if isinstance(source_metadata, dict) and source_metadata.get("subrule"):
        data["subrule"] = _safe_name(source_metadata.get("subrule"))
    if rule and canonical_rule != str(rule):
        data["source_rule"] = str(rule)
    if (out_dir / "answer.png").exists():
        data["answer_image"] = "answer.png"
    details = _compact_source_metadata(source_metadata)
    if metadata_language == "en":
        data = _translate_metadata_to_english(data)
        details = _translate_metadata_to_english(details)
    if details not in ({}, [], None):
        data["details"] = details
    metadata_file = out_dir / "metadata.json"
    metadata_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_file


def _compact_source_metadata(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for k, v in value.items():
            if _is_redundant_path_key(str(k)):
                continue
            compact_value = _compact_source_metadata(v, str(k))
            if compact_value in ({}, [], None):
                continue
            compact[str(k)] = compact_value
        return compact
    if isinstance(value, list):
        items = [_compact_source_metadata(item, key) for item in value]
        return [item for item in items if item not in ({}, [], None)]
    if isinstance(value, str):
        return _compact_string_value(value, key)
    return value


def _is_redundant_path_key(key: str) -> bool:
    normalized = key.lower()
    return normalized in {
        "image",
        "image_path",
        "answer_image_path",
        "content_image_path",
        "choices_image_path",
        "source_dataset",
        "source_json",
        "metadata_file",
        "image_file",
        "question_image",
        "answer_path",
        "output_path",
    }


def _compact_string_value(value: str, key: str | None) -> str | None:
    path_like_key = key is not None and key.lower() in {"cell", "icon", "file", "filename", "path"}
    looks_like_path = "/" in value or "\\" in value
    suffix = Path(value).suffix.lower()
    if path_like_key or looks_like_path:
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
            return Path(value).name
        if looks_like_path:
            return None
    return value




def _attribute_icon_resolver():
    cache: dict[str, Path | None] = {}
    search_roots = [project_root() / "resources" / "attribute", project_root() / "resources", project_root() / "attribute"]

    def resolve(name: Any) -> Path | None:
        if not name:
            return None
        key = str(name)
        if key in cache:
            return cache[key]
        path = Path(key)
        if path.is_absolute() and path.exists():
            cache[key] = path
            return path
        names = [path.name]
        if _is_question_mark_name(path.name):
            names.append("question_mark.png")
        for root in search_roots:
            for name in names:
                hits = list(root.rglob(name))
                if hits:
                    cache[key] = hits[0]
                    return hits[0]
        cache[key] = None
        return None

    return resolve


def _attribute_question_items(title: Any, resolver) -> tuple[list[Any], tuple[int, int], list[Path]]:
    source_files: list[Path] = []
    display_items: list[Any] = []
    grid = (1, 1)
    if isinstance(title, list) and title and all(isinstance(x, list) for x in title):
        rows = len(title)
        cols = max(len(x) for x in title if isinstance(x, list))
        grid = (rows, cols)
        for row in title:
            if not isinstance(row, list):
                continue
            for item in row:
                if _is_question_mark_name(item):
                    display_items.append(QUESTION_MARK_SENTINEL)
                    p = resolver(item)
                    if p:
                        source_files.append(p)
                else:
                    p = resolver(item)
                    if p:
                        display_items.append(p)
                        source_files.append(p)
        return display_items, grid, source_files
    if isinstance(title, list):
        grid = (1, len(title)) if title else (1, 1)
        for item in title:
            if _is_question_mark_name(item):
                display_items.append(QUESTION_MARK_SENTINEL)
                p = resolver(item)
                if p:
                    source_files.append(p)
            else:
                p = resolver(item)
                if p:
                    display_items.append(p)
                    source_files.append(p)
        return display_items, grid, source_files
    return [], grid, []


def _extract_grid_cells(image_path: Path, rows: int, cols: int, *, remove_last: bool = False) -> list[Image.Image]:
    if not image_path.exists():
        return []
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        boxes = _detect_grid_boxes(im, rows, cols)
        if not boxes:
            return []
        cells = [_trim_white_margin(im.crop(box), extra=4) for box in boxes]
        if remove_last and cells:
            cells = cells[:-1]
        return cells


def _extract_spatial_assets(image_path: Path) -> tuple[Image.Image | None, list[Image.Image]]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        lower = im.crop((0, int(h * 0.45), w, h))
        boxes = _detect_grid_boxes(lower, 1, 4)
        option_cells: list[Image.Image] = []
        if boxes:
            option_cells = [_trim_white_margin(lower.crop(box), extra=8) for box in boxes]
        upper = im.crop((0, 0, w, int(h * 0.45) + 20))
        question_panel = _trim_white_margin(upper, extra=12)
        return question_panel, option_cells


def _detect_grid_boxes(im: Image.Image, rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    if rows <= 0 or cols <= 0:
        return []
    v_lines = _detect_line_positions(im, axis="x")
    h_lines = _detect_line_positions(im, axis="y")
    if len(v_lines) >= cols + 1 and len(h_lines) >= rows + 1:
        v_lines = _select_grid_lines(v_lines, cols + 1)
        h_lines = _select_grid_lines(h_lines, rows + 1)
        boxes: list[tuple[int, int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                left = v_lines[c] + 2
                right = v_lines[c + 1] - 2
                top = h_lines[r] + 2
                bottom = h_lines[r + 1] - 2
                if right > left and bottom > top:
                    boxes.append((left, top, right, bottom))
        if len(boxes) == rows * cols:
            return boxes
    bbox = _tight_bbox(im)
    if bbox is None:
        return []
    left, top, right, bottom = bbox
    cell_w = (right - left) / cols
    cell_h = (bottom - top) / rows
    boxes = []
    for r in range(rows):
        for c in range(cols):
            l = int(round(left + c * cell_w))
            t = int(round(top + r * cell_h))
            rr = int(round(left + (c + 1) * cell_w))
            bb = int(round(top + (r + 1) * cell_h))
            boxes.append((l, t, rr, bb))
    return boxes


def _detect_line_positions(im: Image.Image, *, axis: str) -> list[int]:
    gray = im.convert("L")
    threshold = 200
    if np is not None:
        arr = np.array(gray)
        dark = arr < threshold
        if axis == "x":
            counts = dark.sum(axis=0)
            total = dark.shape[0]
        else:
            counts = dark.sum(axis=1)
            total = dark.shape[1]
        positions = [i for i, count in enumerate(counts.tolist()) if count >= total * 0.22]
    else:
        w, h = gray.size
        px = gray.load()
        positions = []
        if axis == "x":
            for x in range(w):
                count = sum(1 for y in range(h) if px[x, y] < threshold)
                if count >= h * 0.22:
                    positions.append(x)
        else:
            for y in range(h):
                count = sum(1 for x in range(w) if px[x, y] < threshold)
                if count >= w * 0.22:
                    positions.append(y)
    return _merge_contiguous_positions(positions)


def _merge_contiguous_positions(positions: list[int]) -> list[int]:
    if not positions:
        return []
    groups: list[list[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos - groups[-1][-1] <= 12:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    return [int(round(sum(group) / len(group))) for group in groups]


def _select_grid_lines(lines: list[int], needed: int) -> list[int]:
    if len(lines) <= needed:
        return lines
    best_combo: tuple[int, ...] | None = None
    best_score: float | None = None
    for combo in combinations(lines, needed):
        gaps = [combo[i + 1] - combo[i] for i in range(len(combo) - 1)]
        if not gaps or min(gaps) <= 8:
            continue
        avg = sum(gaps) / len(gaps)
        var = sum((g - avg) ** 2 for g in gaps) / len(gaps)
        span = combo[-1] - combo[0]
        score = var - span * 0.01
        if best_score is None or score < best_score:
            best_score = score
            best_combo = combo
    if best_combo is not None:
        return list(best_combo)
    return lines[:needed]


def _trim_white_margin(im: Image.Image, *, extra: int = 0) -> Image.Image:
    bg = Image.new(im.mode, im.size, "white")
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return im.copy()
    left, top, right, bottom = bbox
    left = max(0, left - extra)
    top = max(0, top - extra)
    right = min(im.width, right + extra)
    bottom = min(im.height, bottom + extra)
    return im.crop((left, top, right, bottom))


def _tight_bbox(im: Image.Image) -> tuple[int, int, int, int] | None:
    gray = im.convert("L")
    bg = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, bg)
    return diff.getbbox()


def _read_json(path: Path | None) -> Any:
    if not path:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _numeric_images(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.png") if p.stem.isdigit()], key=lambda p: int(p.stem))



def _question_files_from_meta(q_dir: Path, meta: Any) -> list[Path]:
    if not isinstance(meta, dict):
        return []
    value = meta.get("question_image")
    if isinstance(value, list):
        return [q_dir / str(x) for x in value if isinstance(x, (str, int)) and (q_dir / str(x)).exists()]
    return []


def _grid_from_positional_meta(meta: Any, count: int) -> tuple[int, int]:
    if isinstance(meta, dict):
        grid = meta.get("grid_type") or meta.get("grid") or meta.get("grid_size")
        parsed = _grid_from_style_grid(grid, count)
        if count == 4:
            return (1, 4)
        if count == 6:
            return (2, 3)
        if count in (8, 9):
            return (3, 3)
        if parsed[0] * parsed[1] >= count:
            return parsed
    if count == 4:
        return (1, 4)
    if count == 6:
        return (2, 3)
    if count in (8, 9):
        return (3, 3)
    return _auto_grid(count)


def _grid_from_style_grid(grid_value: Any, count: int) -> tuple[int, int]:
    if isinstance(grid_value, list) and len(grid_value) >= 2:
        nums = [x for x in grid_value if isinstance(x, int)]
        if len(nums) >= 2 and nums[0] > 0 and nums[1] > 0:
            cols, rows = nums[0], nums[1]
            if rows * cols >= count:
                return (rows, cols)
            if cols * rows >= count:
                return (cols, rows)
    if count == 9:
        return (3, 3)
    if count == 6:
        return (2, 3)
    if count == 5:
        return (1, 5)
    if count == 4:
        return (1, 4)
    return _auto_grid(count)


def _auto_grid(count: int) -> tuple[int, int]:
    if count <= 0:
        return (1, 1)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def _extract_answer(meta: Any) -> Any:
    if not isinstance(meta, dict):
        return None
    for key in ("answer", "correct_answer", "answer_choice", "ans", "answer_label", "gt"):
        if key in meta:
            return meta[key]
    ans_cells = meta.get("ans_cells")
    if isinstance(ans_cells, list):
        return ans_cells
    return None


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    candidates = [
        font_path,
        str(project_root() / "fonts" / "DejaVuSans.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _coerce_existing_path(value: Any, base: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value)
    if not raw:
        return None
    path = Path(raw)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    candidates.append(base / path)
    candidates.append(project_root() / path)
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else None


def _find_spatial_question_dir(src: Path, qid: str, rule: str | None) -> Path | None:
    candidates: list[Path] = []
    if rule:
        candidates.extend((src / rule).glob(f"*{qid}*"))
    candidates.extend(src.glob(f"**/*{qid}*"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _find_spatial_render(source_dir: Path | None, qid: str) -> Path | None:
    if not source_dir or not source_dir.exists():
        return None
    candidates = list(source_dir.glob(f"*{qid}*.png"))
    candidates.extend(source_dir.glob("puzzle*.png"))
    candidates.extend(source_dir.glob("*.png"))
    if not candidates:
        return None

    def area(path: Path) -> int:
        try:
            with Image.open(path) as im:
                return im.width * im.height
        except Exception:
            return 0

    return max(candidates, key=area)


def _is_question_mark_name(value: Any) -> bool:
    if value is None:
        return False
    return Path(str(value)).name.lower() in ATTRIBUTE_QUESTION_MARK_NAMES


def _looks_like_question_mark_path(path: Path) -> bool:
    return path.name.lower() in ATTRIBUTE_QUESTION_MARK_NAMES or path.stem.lower() in {"q", "question_mark", "question-mark"}


def _option_labels(count: int) -> list[str]:
    return OPTION_LABEL_POOL[:count]


def _raven_answer_label(answer: Any, n_options: int) -> Any:
    if isinstance(answer, int) and 0 <= answer < n_options:
        return _option_labels(n_options)[answer]
    return answer



def _canonical_rule_name(rule: Any) -> str:
    if rule is None:
        return "misc"
    text = str(rule).strip()
    if not text:
        return "misc"
    if text in RULE_NAME_MAP:
        return RULE_NAME_MAP[text]
    lower = text.lower().replace("-", "_").replace(" ", "_")
    if lower == "equal":
        return "xnor"
    if lower in {"and", "or", "xor", "xnor", "line", "curve", "angle", "cart", "space", "parts", "translate", "rotate", "flip", "element", "group", "unfolding", "three_view", "reconstruction_3d", "view_consistency", "multiple_views", "raven", "sudoku"}:
        return lower
    return _safe_name(RULE_NAME_MAP.get(text, text))


def _normalize_numerical_rule(meta: dict[str, Any]) -> str:
    raw = str(meta.get("rule") or meta.get("type") or "").strip()
    text = raw.lower().replace("_", "-")

    def step_rule(prefix: str, chinese: str) -> str | None:
        match = re.search(fr"{chinese}\s*(\d+)", raw)
        if match:
            return f"{prefix}_{match.group(1)}"
        match = re.search(fr"{prefix}[-_\s]*(\d+)", text)
        if match:
            return f"{prefix}_{match.group(1)}"
        return None

    if not raw:
        return "misc"
    for prefix, chinese in (("increasing", "递增"), ("decreasing", "递减")):
        stepped = step_rule(prefix, chinese)
        if stepped:
            return stepped
    if "恒定" in raw or "相同" in raw or "constant" in text or "const" == text:
        return "constant"
    if "恒为偶数" in raw or "均为偶数" in raw or "all-even" in text or "always-even" in text:
        return "always_even"
    if "恒为奇数" in raw or "均为奇数" in raw or "all-odd" in text or "always-odd" in text:
        return "always_odd"
    if "先偶数后奇数" in raw or "even-odd-even-odd" in text or "e-o-e-o" in text:
        return "even_odd_even_odd"
    if "先奇数后偶数" in raw or "odd-even-odd-even" in text or "o-e-o-e" in text:
        return "odd_even_odd_even"
    if "递增" in raw or text.startswith("increasing") or text == "up":
        return "increasing"
    if "递减" in raw or text.startswith("decreasing") or text == "flow":
        return "decreasing"
    cleaned = _safe_name(raw)
    if cleaned.startswith("question_") or len(cleaned) > 48:
        return "misc"
    return cleaned


def _next_canonical_qid(counters: dict[str, int], rule: str | None) -> str:
    key = _safe_name(rule or "default")
    counters[key] = counters.get(key, 0) + 1
    return f"question{counters[key]}"


def canonical_category_name(module: str) -> str:
    return CANONICAL_CATEGORY_MAP.get(module, _safe_name(module))


def _canonical_category_name(module: str) -> str:
    return canonical_category_name(module)


def _public_module_index_stem(module: str) -> str:
    return PUBLIC_MODULE_INDEX_MAP.get(module, _safe_name(module))


def _canonical_question_dir_name(module: str, rule: str | None, qid: Any) -> Path:
    qname = _canonical_question_leaf(qid)
    if rule:
        return Path(_safe_name(rule), qname)
    return Path(qname)


def _canonical_question_leaf(qid: Any) -> str:
    raw = str(qid).strip()
    if not raw:
        return "question1"
    m = re.fullmatch(r"question[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return f"question{max(1, int(m.group(1)))}"
    m = re.fullmatch(r"task[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return f"question{int(m.group(1)) + 1}"
    if re.fullmatch(r"\d+", raw):
        return f"question{max(1, int(raw))}"
    m = re.search(r"(?:^|_)(\d+)$", raw)
    if m:
        return f"question{int(m.group(1)) + 1 if int(m.group(1)) == 0 else int(m.group(1))}"
    return _safe_name(raw)


def _clear_previous_module_outputs(module: str, normalized_root: Path) -> None:
    ensure_dir(normalized_root)
    first_module_for_category = {category: category for category in CANONICAL_CATEGORY_ORDER}
    category = canonical_category_name(module)
    if module != first_module_for_category.get(category):
        return
    for child in list(normalized_root.iterdir()):
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()



def _write_canonical_category_indexes(normalized_root: Path, module_manifest: dict[str, Any]) -> list[Path]:
    module_name = str(module_manifest.get("module"))
    category = str(module_manifest.get("category") or normalized_root.name)
    category_index_path = normalized_root / "index.json"
    if category_index_path.exists():
        try:
            category_index = json.loads(category_index_path.read_text(encoding="utf-8"))
        except Exception:
            category_index = {}
    else:
        category_index = {}

    existing_questions = [
        record
        for record in category_index.get("questions", []) if isinstance(category_index, dict)
        if isinstance(record, dict)
    ]

    new_questions = [_dataset_index_record(question, normalized_root, category) for question in module_manifest.get("questions", [])]
    questions = sorted(existing_questions + new_questions, key=lambda item: item.get("id", ""))
    category_index = {
        "category": category,
        "count": len(questions),
        "questions": questions,
    }
    warnings_list = module_manifest.get("warnings") or []
    if warnings_list:
        category_index["warnings"] = warnings_list
    category_index_path.write_text(json.dumps(category_index, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in normalized_root.glob("*_index.json"):
        path.unlink(missing_ok=True)
    return [category_index_path]


def _dataset_index_record(question: dict[str, Any] | NormalizedQuestion, category_root: Path, category: str) -> dict[str, Any]:
    if isinstance(question, NormalizedQuestion):
        question_dir = _relpath(question.output_dir, category_root) or question.output_dir.name
        metadata_rel = _relpath(question.metadata_file, category_root) if question.metadata_file else f"{question_dir}/metadata.json"
    else:
        output_dir = question.get("output_dir", "")
        question_dir = str(output_dir).split(f"{category}/", 1)[-1] if output_dir else ""
        metadata_rel = f"{question_dir}/{question.get('metadata', 'metadata.json')}"
    metadata_path = category_root / metadata_rel
    metadata = _read_json(metadata_path) if metadata_path.exists() else {}
    stem_images = [f"{question_dir}/{path}" for path in metadata.get("stem_images", [])]
    option_images = [f"{question_dir}/{path}" for path in metadata.get("option_images", [])]
    record = {
        "id": f"{category}/{question_dir}",
        "category": category,
        "rule": metadata.get("rule"),
        "question_dir": question_dir,
        "question_image": f"{question_dir}/{metadata.get('question_image', QUESTION_IMAGE_NAME)}",
        "metadata": metadata_rel,
        "answer": metadata.get("answer"),
        "stem_images": stem_images,
        "option_images": option_images,
    }
    if metadata.get("subrule"):
        record["subrule"] = metadata.get("subrule")
    if metadata.get("answer_image"):
        record["answer_image"] = f"{question_dir}/{metadata['answer_image']}"
    return record


def write_global_index(output_root: str | Path) -> Path:
    root = Path(output_root)
    categories: dict[str, dict[str, Any]] = {}
    questions: list[dict[str, Any]] = []
    for category in CANONICAL_CATEGORY_ORDER:
        category_index_path = root / category / "index.json"
        if not category_index_path.exists():
            ensure_dir(category_index_path.parent)
            empty_index = {"category": category, "count": 0, "questions": []}
            category_index_path.write_text(json.dumps(empty_index, ensure_ascii=False, indent=2), encoding="utf-8")
            categories[category] = {"count": 0, "index": f"{category}/index.json"}
            continue
        category_index = _read_json(category_index_path)
        category_questions = category_index.get("questions", []) if isinstance(category_index, dict) else []
        categories[category] = {"count": len(category_questions), "index": f"{category}/index.json"}
        for record in category_questions:
            if not isinstance(record, dict):
                continue
            prefixed = dict(record)
            for key in ("question_dir", "question_image", "metadata", "answer_image"):
                if prefixed.get(key):
                    prefixed[key] = f"{category}/{prefixed[key]}"
            prefixed["stem_images"] = [f"{category}/{path}" for path in prefixed.get("stem_images", [])]
            prefixed["option_images"] = [f"{category}/{path}" for path in prefixed.get("option_images", [])]
            questions.append(prefixed)
    data = {
        "count": len(questions),
        "categories": categories,
        "questions": questions,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return index_path



def _extract_raven_grid_cells(image_path: Path, rows: int, cols: int, *, remove_last: bool = False) -> list[Image.Image]:
    if not image_path.exists():
        return []
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        padding = 12
        title_h = 34
        cell = int(round((im.width - (cols + 1) * padding) / cols))
        if cell <= 0:
            return _extract_grid_cells(image_path, rows, cols, remove_last=remove_last)
        cells: list[Image.Image] = []
        for r in range(rows):
            for c in range(cols):
                x0 = padding + c * (cell + padding)
                y0 = title_h + padding + r * (cell + padding)
                box = (x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2)
                cells.append(_trim_white_margin(im.crop(box), extra=4))
        if remove_last and cells:
            cells = cells[:-1]
        return cells

def _safe_name(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "item"
    text = _english_rule_name(text)
    text = re.sub(r"[^A-Za-z0-9_.\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "item"


def _english_rule_name(text: str) -> str:
    if text in RULE_NAME_MAP:
        return RULE_NAME_MAP[text]
    result = text
    for cn, en in sorted(RULE_NAME_MAP.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(cn, f"_{en}_")
    for cn, en in CHINESE_CHAR_NAME_MAP.items():
        result = result.replace(cn, f"_{en}_")
    return result


def _natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def _relpath(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _option_images(directory: Path) -> list[Path]:
    files = []
    for label in OPTION_LABEL_POOL[:8]:
        p = directory / f"{label}.png"
        if p.exists():
            files.append(p)
    return files
