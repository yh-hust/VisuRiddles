# VisuRiddles Data Synthesis Engine

This repository integrates seven visual-reasoning data synthesizers into one unified generation pipeline.

## Public categories

The public category names follow the project taxonomy:

- `attribute`: element/group attribute reasoning.
- `positional`: positional reasoning over translation, rotation, and flipping.
- `spatial`: 3D spatial reasoning.
- `numerical`: numerosity and quantity-pattern reasoning.
- `stylistic`: logical style operations.
- `raven`: RAVEN-style abstract matrix reasoning.
- `sudoku`: constraint-based Sudoku reasoning.

## Subcategories

The synthesizer exposes one Python module per subcategory:

```text
attribute/element.py
attribute/group.py
positional/translate.py
positional/rotate.py
positional/flip.py
spatial/unfolding.py
spatial/three_view.py
spatial/reconstruction_3d.py
spatial/view_consistency.py
spatial/multiple_views.py
numerical/line.py
numerical/curve.py
numerical/angle.py
numerical/cart.py
numerical/space.py
numerical/parts.py
stylistic/and.py
stylistic/or.py
stylistic/xor.py
stylistic/xnor.py
raven/raven.py
sudoku/sudoku.py
```


## Internal code organization

The category folders are the canonical implementation locations. Obsolete helper trees such as `numerical/num_gen`, `numerical/num_gen2`, `positional/generator`, `raven/engine.py`, `sudoku/engine.py`, and `spatial/generate_spatial_puzzles.py` have been removed from the formal code path. Each paper-defined subcategory has its own generation module. Category-level helper modules are kept only when they are part of the active implementation.

## Installation

```bash
pip install -e ".[dev]"
```

## Generate all categories

```bash
python generate.py \
  --attribute 10 \
  --positional 10 \
  --spatial 10 \
  --numerical 10 \
  --stylistic 10 \
  --raven 10 \
  --sudoku 10 \
  --out_root ./outputs
```

Metadata is written in Chinese by default. To generate English-only `metadata.json` and `index.json` files, add:

```bash
--metadata_language en
```

Example:

```bash
python generate.py \
  --attribute 1 \
  --positional 1 \
  --spatial 1 \
  --numerical 1 \
  --stylistic 1 \
  --raven 1 \
  --sudoku 1 \
  --metadata_language en \
  --out_root ./outputs_en
```

The same all-category entrypoint is also available at:

```bash
python scripts/generate.py --attribute 1 --positional 1 --spatial 1 --numerical 1 --stylistic 1 --raven 1 --sudoku 1 --out_root ./outputs
```

## Generate one category with per-subcategory counts

Examples:

```bash
python scripts/generate_positional.py --translate 10 --rotate 10 --flip 10 --out_root ./outputs
python scripts/generate_spatial.py --unfolding 5 --three_view 5 --reconstruction_3d 5 --view_consistency 5 --multiple_views 5 --out_root ./outputs
python scripts/generate_numerical.py --line 5 --curve 5 --angle 5 --cart 5 --space 5 --parts 5 --out_root ./outputs
python scripts/generate_stylistic.py --and 5 --or 5 --xor 5 --xnor 5 --out_root ./outputs
python scripts/generate_attribute.py --element 5 --group 5 --out_root ./outputs
python scripts/generate_raven.py --raven 5 --out_root ./outputs
python scripts/generate_sudoku.py --sudoku 5 --out_root ./outputs

# English metadata for a single category
python scripts/generate_stylistic.py --and 5 --metadata_language en --out_root ./outputs_en
```

## Output format

The final dataset is written directly under `out_root`:

```text
outputs/
  attribute/
  positional/
  spatial/
  numerical/
  stylistic/
  raven/
  sudoku/
  index.json
```

Each question folder contains:

```text
question.png
metadata.json
subimages/
  stem/
  options/
```

Category-level `index.json` files and the root `index.json` provide compact dataset records, including question image paths, metadata paths, rule names, answers, stem images, and option images.

Raw intermediate outputs are temporary and are deleted automatically after normalization.

## Tests

```bash
pytest -q
```
