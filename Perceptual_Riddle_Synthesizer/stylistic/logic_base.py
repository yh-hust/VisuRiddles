import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from tqdm import tqdm
from PIL import Image
from matplotlib.transforms import Bbox

DEFAULT_PARAMS = {
    "randomization_cfg": {
        "main_image": {
            "dpi_range": [30, 200],
            "h_pad_range_in": [0, 2],
            "v_pad_range_in": [0, 2],
            "h_no_padding_prob": 0.8,
            "v_no_padding_prob": 0.8
        },
        "sub_image": {
            "dpi_range": [100, 100],
            "pad_range_in": [0, 0]
        }
    },
    "hatch_map": {
        '//': '斜线', '///': '斜线', '////': '斜线', '/////': '斜线',
        '\\': '反斜线', "\\\\": '反斜线', "\\\\\\": '反斜线', "\\\\\\\\": '反斜线',
        '||': '竖线', '|||': '竖线', '||||': '竖线',
        '--': '横线', '---': '横线', '----': '横线',
        '+': '网格', '++': '网格', '+++': '网格',
        'x': '交叉网格', 'xx': '交叉网格', 'xxx': '交叉网格',
        'o': '小圆点', 'oo': '小圆点',
        'O': '大圆点', 'OO': '大圆点',
        '.': '点状', '..': '点状', '...': '点状',
        '*': '星号', '**': '星号',
        'gray': '灰色', 'black': '黑色',
    },
    "shapes": ['grid3', 'grid4', 'fan4', 'fan6', 'fan8', 'circleblock', 'cross', 'hexagon', 'hexagon12'],
    "rules": ['xor', 'and', 'or', 'equal'],
    "linewidth_options": [0.5, 1.0, 1.5, 2.0, 2.5],
    "hatch_linewidth_options": [0.5, 1.0, 1.5, 2.0]
}

class PatternTile:
    def generate(self):
        raise NotImplementedError
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        raise NotImplementedError
    def serialize(self, data):
        raise NotImplementedError

class TextureStrategy:
    def apply(self, facecolor, value):
        raise NotImplementedError

class FixedHatchTexture(TextureStrategy):
    def __init__(self, pattern='/'):
        self.pattern = pattern
    def apply(self, facecolor, value):
        return (facecolor, self.pattern if value else None)

class SolidFillTexture(TextureStrategy):
    def __init__(self, black_color='black', white_color='white'):
        self.black_color = black_color
        self.white_color = white_color
    def apply(self, facecolor, value):
        return ((self.black_color if value else self.white_color), None)

class XorRule:
    def apply(self, a, b): return np.bitwise_xor(a, b)
class AndRule:
    def apply(self, a, b): return np.bitwise_and(a, b)
class OrRule:
    def apply(self, a, b): return np.bitwise_or(a, b)
class EqualRule:
    def apply(self, a, b): return (a == b).astype(int)

def get_rule(name='xor'):
    if name == 'xor':   return XorRule()
    if name == 'and':   return AndRule()
    if name == 'or':    return OrRule()
    if name == 'equal': return EqualRule()
    raise ValueError(f"Unknown rule type: {name}")

def combine_rule(**rule_kwargs):
    candidates = [v for k, v in rule_kwargs.items() if k.startswith('rule_') and v]
    choice = random.choice(candidates) if candidates else random.choice(DEFAULT_PARAMS['rules'])
    return get_rule(choice), choice

class GridTile(PatternTile):
    def __init__(self, size=3): self.size = size
    def generate(self): return np.random.randint(0, 2, (self.size, self.size))
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        cell = 0.75 * box_size / self.size
        sx, sy = x + 0.125 * box_size, y + 0.125 * box_size
        for i in range(self.size):
            for j in range(self.size):
                fc, hatch = texture_strategy.apply('white', data[i, j])
                ax.add_patch(patches.Rectangle(
                    (sx + j*cell, sy + i*cell), cell, cell,
                    facecolor=fc, edgecolor='black', hatch=hatch, linewidth=edge_lw
                ))
    def serialize(self, data):
        return np.flipud(data).flatten(order='C').tolist()

class FanTile(PatternTile):
    def __init__(self, num_sectors=8):
        self.num_sectors = num_sectors
    def generate(self):
        return np.random.randint(0, 2, self.num_sectors)
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        r = 0.75 * box_size / 2
        th = 360 / self.num_sectors
        cx, cy = x + box_size / 2, y + box_size / 2
        for i in range(self.num_sectors):
            fc, hatch = texture_strategy.apply('white', data[i])
            ax.add_patch(patches.Wedge(
                (cx, cy), r, i * th, (i + 1) * th,
                facecolor=fc, edgecolor='black', hatch=hatch, linewidth=edge_lw
            ))
    def serialize(self, data):
        n, th = self.num_sectors, 360.0 / self.num_sectors
        centers = [(i + 0.5) * th for i in range(n)]
        order = sorted(range(n), key=lambda i: (90 - centers[i]) % 360)
        return [int(data[i]) for i in order]

class CircleBlockTile(PatternTile):
    def __init__(self): self.rows, self.cols = 2,2
    def generate(self): return np.random.randint(0, 2, (self.rows, self.cols))
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        cs = box_size/self.rows
        for i in range(self.rows):
            for j in range(self.cols):
                fc, hatch = texture_strategy.apply('white', data[i,j])
                cx = x + j*cs + cs/2; cy = y + (self.rows-1-i)*cs + cs/2
                ax.add_patch(patches.Circle(
                    (cx, cy), radius=cs/2.5,
                    facecolor=fc, edgecolor='black', hatch=hatch, linewidth=edge_lw
                ))
    def serialize(self, data):
        return [int(data[i,j]) for i in range(self.rows) for j in range(self.cols)]

class CrossTile(PatternTile):
    def __init__(self): self.num_regions = 8
    def generate(self): return np.random.randint(0, 2, self.num_regions)
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        B, off = 0.85*box_size, 0.075*box_size
        x0, y0 = x+off, y+off; cx, cy = x0+B/2, y0+B/2
        pts = [
            [(x0, y0+B), (x0+B/2, y0+B), (cx,cy)],[(x0,y0+B/2),(x0,y0+B),(cx,cy)],
            [(x0+B/2,y0+B),(x0+B,y0+B),(cx,cy)],[(x0+B,y0+B),(x0+B,y0+B/2),(cx,cy)],
            [(x0+B,y0+B/2),(x0+B,y0),(cx,cy)],[(x0+B,y0),(x0+B/2,y0),(cx,cy)],
            [(x0+B/2,y0),(x0,y0),(cx,cy)],[(x0,y0),(x0,y0+B/2),(cx,cy)]
        ]
        for i, poly in enumerate(pts):
            fc, hatch = texture_strategy.apply('white', data[i])
            ax.add_patch(patches.Polygon(
                poly, facecolor=fc, edgecolor='black', hatch=hatch, linewidth=edge_lw
            ))
    def serialize(self, data):
        box, off = 0.85, 0.075; x0, y0 = off, off; cx, cy = x0+box/2, y0+box/2
        pts = [
            [(x0, y0+box), (x0+box/2, y0+box), (cx,cy)],[(x0,y0+box/2),(x0,y0+box),(cx,cy)],
            [(x0+box/2,y0+box),(x0+box,y0+box),(cx,cy)],[(x0+box,y0+box),(x0+box,y0+box/2),(cx,cy)],
            [(x0+box,y0+box/2),(x0+box,y0),(cx,cy)],[(x0+box,y0),(x0+box/2,y0),(cx,cy)],
            [(x0+box/2,y0),(x0,y0),(cx,cy)],
            [(x0,y0),(x0,y0+box/2),(cx,cy)]
        ]
        angles = []
        for poly in pts:
            mx = sum(p[0] for p in poly)/3; my = sum(p[1] for p in poly)/3
            angles.append((math.degrees(math.atan2(my-cy, mx-cx))+360)%360)
        order = sorted(range(self.num_regions), key=lambda i: (90-angles[i])%360)
        return [int(data[i]) for i in order]

class HexagonTile12(PatternTile):
    def __init__(self): self.num_regions = 12
    def generate(self): return np.random.randint(0, 2, self.num_regions)
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        cx, cy = x + box_size/2, y + box_size/2
        radius = 0.75 * box_size / 2
        verts = []
        for i in range(6):
            theta = np.deg2rad(-60*i + 30)
            verts.append((cx + radius*math.cos(theta), cy + radius*math.sin(theta)))
        mids = []
        for i in range(6):
            p1, p2 = verts[i], verts[(i+1)%6]
            mids.append(((p1[0]+p2[0])/2, (p1[1]+p2[1])/2))
        boundary = []
        for i in range(6):
            boundary.append(verts[i])
            boundary.append(mids[i])
        start_index = 10
        boundary = boundary[start_index:] + boundary[:start_index]
        for i in range(12):
            p1 = boundary[i]
            p2 = boundary[(i+1)%12]
            fc, hatch = texture_strategy.apply('white', data[i])
            ax.add_patch(patches.Polygon(
                [p1, p2, (cx, cy)], facecolor=fc, edgecolor='black', hatch=hatch, linewidth=edge_lw
            ))
        ax.add_patch(patches.Polygon(verts, closed=True, fill=False, edgecolor='black', linewidth=edge_lw))
    def serialize(self, data): return [int(data[i]) for i in range(12)]

class HexagonTile(PatternTile):
    def __init__(self): self.num_regions = 10
    def generate(self): return np.random.randint(0, 2, self.num_regions)
    def draw(self, ax, x, y, data, box_size, texture_strategy, edge_lw):
        cx, cy = x + box_size / 2, y + box_size / 2
        r = 0.9 * box_size / 2
        verts = [(cx + r * math.cos(math.radians(60*i)), cy + r * math.sin(math.radians(60*i))) for i in range(6)]
        center_point = (cx, cy)
        I_right = (verts[1][0], cy)
        I_left = (verts[2][0], cy)
        ten_final_regions = [
            [verts[2], verts[3], I_left], [verts[2], I_left, center_point], [verts[1], verts[2], center_point],
            [verts[1], I_right, center_point], [verts[0], verts[1], I_right], [verts[3], verts[4], I_left],
            [verts[4], I_left, center_point], [verts[4], verts[5], center_point], [verts[5], I_right, center_point],
            [verts[5], verts[0], I_right],
        ]
        for idx, region_verts in enumerate(ten_final_regions):
            region_data = data[idx]
            fc, hatch = texture_strategy.apply('white', region_data)
            ax.add_patch(patches.Polygon(region_verts, facecolor=fc, edgecolor='black', hatch=hatch, linewidth=0.5))
        main_lines_lw = edge_lw
        for i in range(3):
            ax.add_line(plt.Line2D([verts[i][0], verts[i+3][0]], [verts[i][1], verts[i+3][1]], color='black', linewidth=main_lines_lw))
        ax.add_line(plt.Line2D([verts[1][0], verts[5][0]], [verts[1][1], verts[5][1]], color='black', linewidth=main_lines_lw))
        ax.add_line(plt.Line2D([verts[2][0], verts[4][0]], [verts[2][1], verts[4][1]], color='black', linewidth=main_lines_lw))
        ax.add_patch(patches.Polygon(verts, closed=True, fill=False, edgecolor='black', linewidth=main_lines_lw + 0.5))
    def serialize(self, data): return [int(data[i]) for i in range(10)]

def _generate_puzzles_internal(params, mixed_mode):
    cfg = params['generate_cfg']
    rand_cfg = params.get("randomization_cfg", {})

    num = cfg['num']
    output_dir = cfg['output_dir']
    json_path = cfg['json_path']
    os.makedirs(output_dir, exist_ok=True)

    hatch_map = params['hatch_map']
    shapes = params['shapes']
    tile_map = {
        'grid3': GridTile(3), 'grid4': GridTile(4), 'fan4': FanTile(4),
        'fan6': FanTile(6), 'fan8': FanTile(8), 'circleblock': CircleBlockTile(),
        'cross': CrossTile(), 'hexagon': HexagonTile(), 'hexagon12': HexagonTile12()
    }
    name_cn = {
        'grid3': '3×3正方形', 'grid4': '4×4正方形', 'fan4': '扇形(4)',
        'fan6': '扇形(6)', 'fan8': '扇形(8)', 'circleblock': '圆形',
        'cross': '十字型', 'hexagon': '六边形(10区)', 'hexagon12': '六边形(12区)'
    }
    rule_cn = {'xor':'异或','and':'与','or':'或','equal':'同或'}

    metadata_list = []
    for idx in tqdm(range(1, num + 1), desc="Generating Stylistic logical Puzzles"):
        tex = random.choice(list(hatch_map.keys()))
        lw  = random.choice(params['linewidth_options'])
        hlw = random.choice(params['hatch_linewidth_options'])

        with plt.rc_context({'hatch.linewidth': hlw}):
            if tex == 'gray':
                gray = str(random.uniform(0.3,0.7))
                strategy = SolidFillTexture(black_color=gray, white_color='white')
            else:
                strategy = (SolidFillTexture(black_color=tex, white_color='white')
                            if tex=='black' else FixedHatchTexture(tex))

            forced_rule = cfg.get('rule_name')
            rk = forced_rule or None
            rule = get_rule(rk) if rk else None
            if rule is None:
                rule, rk = combine_rule(rule_1=None, rule_2=None)
            mix = (mixed_mode if isinstance(mixed_mode, bool)
                   else random.random()<float(mixed_mode))
            keys = random.sample(shapes, 3) if mix else [random.choice(shapes)]*3

            rows = []
            for r in range(3):
                a = tile_map[keys[r]].generate()
                b = tile_map[keys[r]].generate()
                if r < 2:
                    rows.append([a, b, rule.apply(a,b)])
                else:
                    correct = rule.apply(a,b)
                    rows.append([a, b, None])

            wrongs = []
            while len(wrongs) < 3:
                w = tile_map[keys[2]].generate()
                if all(not np.array_equal(w, x) for x in wrongs) and not np.array_equal(w, correct):
                    wrongs.append(w)
            choices = wrongs + [correct]
            random.shuffle(choices)
            for i,ch in enumerate(choices):
                if np.array_equal(ch, correct):
                    correct_label = chr(ord('A') + i)

            sub_cfg = rand_cfg.get("sub_image", {})
            sub_dpi_range = sub_cfg.get("dpi_range", [100, 200])
            sub_pad_range = sub_cfg.get("pad_range_in", [0.05, 0.2])
            subdir = os.path.join(output_dir, 'subimgs', str(idx))
            os.makedirs(subdir, exist_ok=True)

            img_idx = 1
            for col in range(3):
                for row in range(3):
                    data = rows[row][col] if rows[row][col] is not None else correct
                    fig_s, ax_s = plt.subplots(figsize=(2,2))
                    ax_s.set_xlim(0,1.5); ax_s.set_ylim(0,1.5); ax_s.axis('off')
                    tile_map[keys[row]].draw(ax_s, 0, 0, data, 1.5, strategy, lw)
                    sub_dpi = random.randint(*sub_dpi_range)
                    sub_pad = random.uniform(*sub_pad_range)
                    fig_s.savefig(os.path.join(subdir, f"{img_idx}.png"),
                                  bbox_inches='tight', dpi=sub_dpi, pad_inches=sub_pad)
                    plt.close(fig_s); img_idx += 1

            for i,ch in enumerate(choices):
                fig_s, ax_s = plt.subplots(figsize=(2,2))
                ax_s.set_xlim(0,1.5); ax_s.set_ylim(0,1.5); ax_s.axis('off')
                tile_map[keys[2]].draw(ax_s, 0,0, ch, 1.5, strategy, lw)
                sub_dpi = random.randint(*sub_dpi_range)
                sub_pad = random.uniform(*sub_pad_range)
                fig_s.savefig(os.path.join(subdir, f"{chr(ord('A')+i)}.png"),
                              bbox_inches='tight', dpi=sub_dpi, pad_inches=sub_pad)
                plt.close(fig_s)

            fig, ax = plt.subplots(figsize=(10,12))
            ax.set_xlim(0,7); ax.set_ylim(-0.3,7.2)
            ax.set_aspect('equal'); ax.axis('off')
            bs, ox, oy = 1.5, 1.25, 2.2
            for r in range(3):
                for c in range(3):
                    x, y = ox+c*bs, oy+(2-r)*bs
                    ax.add_patch(patches.Rectangle((x,y), bs, bs, linewidth=lw, edgecolor='black', facecolor='white'))
                    cell = rows[r][c]
                    if cell is None:
                        ax.text(x+bs/2, y+bs/2, '?', fontsize=70, ha='center', va='center')
                    else:
                        tile_map[keys[r]].draw(ax, x, y, cell, bs, strategy, lw)
            for i,ch in enumerate(choices):
                x, y = ox+i*bs-0.75, 0.3
                ax.add_patch(patches.Rectangle((x,y), bs, bs, linewidth=lw, edgecolor='black', facecolor='white'))
                tile_map[keys[2]].draw(ax, x, y, ch, bs, strategy, lw)
                ax.text(x+bs/2, y-0.2, chr(ord('A')+i), ha='center', va='top', fontsize=14)

            main_cfg = rand_cfg.get("main_image", {})
            main_dpi_range = main_cfg.get("dpi_range", [250, 450])
            h_pad_range = main_cfg.get("h_pad_range_in", [0.4, 1.2])
            v_pad_range = main_cfg.get("v_pad_range_in", [0.4, 1.2])
            h_no_pad_prob = main_cfg.get("h_no_padding_prob", 0.0)
            v_no_pad_prob = main_cfg.get("v_no_padding_prob", 0.0)
            random_horizontal_pad_in = 0 if random.random() < h_no_pad_prob else random.uniform(*h_pad_range)
            random_vertical_pad_in = 0 if random.random() < v_no_pad_prob else random.uniform(*v_pad_range)

            img_name = f"{idx}.png"
            output_filepath = os.path.join(output_dir, img_name)
            random_dpi = random.randint(*main_dpi_range)

            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            tight_bbox_px = ax.get_tightbbox(renderer)
            tight_bbox_in = tight_bbox_px.transformed(fig.dpi_scale_trans.inverted())
            final_bbox_in = Bbox.from_bounds(
                tight_bbox_in.x0 - random_horizontal_pad_in,
                tight_bbox_in.y0 - random_vertical_pad_in,
                tight_bbox_in.width + 2 * random_horizontal_pad_in,
                tight_bbox_in.height + 2 * random_vertical_pad_in
            )
            plt.savefig(output_filepath, dpi=random_dpi, bbox_inches=final_bbox_in)
            plt.close(fig)

            cells_meta = []
            for r in range(3):
                for c in range(3):
                    cell = rows[r][c]
                    if 'correct' not in locals():
                        last_row_a = tile_map[keys[2]].generate()
                        last_row_b = tile_map[keys[2]].generate()
                        correct = rule.apply(last_row_a, last_row_b)
                    length = correct.size
                    val = (tile_map[keys[r]].serialize(cell) if cell is not None else [-1]*length)
                    cells_meta.append({"type": name_cn[keys[r]], "value": val})
            ans_cells_meta = []
            for ch in choices:
                ans_cells_meta.append({"type": name_cn[keys[2]], "value": tile_map[keys[2]].serialize(ch)})

            metadata_list.append({
                "image": img_name, "type": "黑白运算", "grid": [3,3],
                "cells": cells_meta, "纹理": hatch_map[tex], "rule": rule_cn[rk],
                "answer": correct_label, "ans_cells": ans_cells_meta
            })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

def generate_logic_tasks(num, out_dir, mixed_mode=0.5, rule_name=None):
    print("--- Starting Stylistic logical question generation ---")
    print(f"Question count: {num}, output directory: {out_dir}")
    import copy
    params = copy.deepcopy(DEFAULT_PARAMS)
    params['generate_cfg'] = {
        "num": num,
        "output_dir": out_dir,
        "json_path": os.path.join(out_dir, 'metadata.json'),
        "rule_name": rule_name,
    }
    _generate_puzzles_internal(params, mixed_mode)
    print("--- Stylistic logical question generation completed ---")

if __name__ == '__main__':
    print("Running gen2.py as a standalone script with default parameters...")
    generate_logic_tasks(num=10, out_dir='./output_stylistic_logic_test')
