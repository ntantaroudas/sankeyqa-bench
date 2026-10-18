"""Draw a career-map Sankey diagram from a flow table with matplotlib.

The drawing follows the layout of the career maps in XR-CareerAssist (four columns for now, 2, 5 and 10 years,
node bars whose height is proportional to the number of profiles, curved ribbons whose width is proportional to the
number of profiles moving between two nodes, labels beside the nodes). Every visual setting that the robustness
study varies is a field of ``RenderStyle``, and the same table always produces the same picture for the same style.
"""
from __future__ import annotations

import dataclasses
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

from .cohort import Cohort, STAGES, LADDER, LATERAL, OTHER

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5fd6", "#d6457a", "#4fb3d9", "#8a8a2a", "#6b6b6b"]
MONO = ["#1f4e8c", "#2a6fbf", "#3c8ad9", "#5aa2e6", "#7fb8ee", "#a3cdf4", "#c6e0f8", "#9bb0c4", "#6b7a8a"]


@dataclasses.dataclass
class RenderStyle:
    """The visual settings of a drawing. ``value_labels`` is "none" (labels only), "nodes" (the number of profiles
    after each label) or "all" (also the number on every ribbon)."""
    background: str = "white"          # "white" or "transparent"
    font_scale: float = 1.0            # 0.7 small, 1.0 normal, 1.4 large
    value_labels: str = "nodes"        # "none", "nodes", "all"
    palette: str = "categorical"       # "categorical" or "mono"
    long_side_px: int = 1024           # 512, 768, 1024, 1536
    show_title: bool = True
    node_order: str = "size"           # "size" (largest at the top) or "fixed" (ladder order)
    ribbon_alpha: float = 0.55
    name: str = "default"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def blend(color: str, a: float) -> str:
    """Opaque colour that looks like ``color`` at opacity ``a`` on white."""
    return to_hex(tuple(1 - a + a * c for c in to_rgb(color)))


def title_for(c: Cohort) -> str:
    cfg = c.config
    kind = "Role evolution" if cfg.map_type == "role" else "Industry shifts"
    return (f"{kind} for a {cfg.start_role} with {cfg.years_experience} years of experience in {cfg.industry}, "
            f"age {cfg.age_group}, target {cfg.target_role} (n = {c.n_profiles} profiles)")


def _order(labels: dict[str, int], how: str) -> list[str]:
    if how == "fixed":
        fixed = LADDER + LATERAL + [OTHER]
        known = [l for l in fixed if l in labels]
        rest = sorted([l for l in labels if l not in fixed])
        return known + rest
    return [k for k, _ in sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))]


def layout(c: Cohort, style: RenderStyle) -> dict[str, Any]:
    """Positions of nodes and ribbons in a unit square (y grows upwards)."""
    n = c.n_profiles
    gap, h_min = 0.035, 0.02
    cols = []
    for s in range(4):
        labels = _order(c.nodes[s], style.node_order)
        usable = 1.0 - gap * (len(labels) - 1)
        # heights proportional to the counts, but no node thinner than h_min so that its label stays legible,
        # the other nodes shrink to make room
        raw = {lab: usable * c.nodes[s][lab] / n for lab in labels}
        small = [lab for lab in labels if raw[lab] < h_min]
        big_total = sum(raw[lab] for lab in labels if lab not in small)
        room = usable - h_min * len(small)
        hs = {lab: (h_min if lab in small else raw[lab] * room / big_total) for lab in labels}
        y = 1.0
        pos = {}
        for lab in labels:
            h = hs[lab]
            pos[lab] = (y - h, y)   # bottom, top
            y -= h + gap
        cols.append(pos)
    # ribbons: outgoing stacked by target position (top first), incoming stacked by source position
    ribbons = []
    for s in range(3):
        src_cursor = {lab: cols[s][lab][1] for lab in cols[s]}
        tgt_cursor = {lab: cols[s + 1][lab][1] for lab in cols[s + 1]}
        order = sorted([(s_, a, b, cnt) for s_, a, b, cnt in c.links if s_ == s],
                       key=lambda l: (-cols[s][l[1]][1], -cols[s + 1][l[2]][1]))
        # outgoing order per source: targets from top to bottom
        out_by_src: dict[str, list] = {}
        for l in order:
            out_by_src.setdefault(l[1], []).append(l)
        in_by_tgt: dict[str, list] = {}
        for l in sorted(order, key=lambda l: (-cols[s + 1][l[2]][1], -cols[s][l[1]][1])):
            in_by_tgt.setdefault(l[2], []).append(l)
        usable_src = {lab: cols[s][lab][1] - cols[s][lab][0] for lab in cols[s]}
        usable_tgt = {lab: cols[s + 1][lab][1] - cols[s + 1][lab][0] for lab in cols[s + 1]}
        src_slot, tgt_slot = {}, {}
        for a, ls in out_by_src.items():
            tot = sum(l[3] for l in ls)
            for l in ls:
                h = usable_src[a] * l[3] / tot
                top = src_cursor[a]
                src_slot[(a, l[2])] = (top - h, top)
                src_cursor[a] = top - h
        for b, ls in in_by_tgt.items():
            tot = sum(l[3] for l in ls)
            for l in ls:
                h = usable_tgt[b] * l[3] / tot
                top = tgt_cursor[b]
                tgt_slot[(l[1], b)] = (top - h, top)
                tgt_cursor[b] = top - h
        for _, a, b, cnt in order:
            ribbons.append({"stage": s, "source": a, "target": b, "count": cnt,
                            "src": src_slot[(a, b)], "tgt": tgt_slot[(a, b)]})
    return {"cols": cols, "ribbons": ribbons}


def _ribbon_path(x0, x1, s_bot, s_top, t_bot, t_top) -> Path:
    cx = (x1 - x0) * 0.5
    verts = [(x0, s_top), (x0 + cx, s_top), (x1 - cx, t_top), (x1, t_top),
             (x1, t_bot), (x1 - cx, t_bot), (x0 + cx, s_bot), (x0, s_bot), (x0, s_top)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
    return Path(verts, codes)


def render(c: Cohort, style: RenderStyle | None = None) -> Image.Image:
    """Draw the diagram and return it as a PIL image (RGB on white, or RGBA when the background is transparent)."""
    style = style or RenderStyle()
    L = layout(c, style)
    cols, ribbons = L["cols"], L["ribbons"]
    fs = 11.0 * style.font_scale
    w_in, h_in = 10.0, 6.0
    dpi = style.long_side_px / w_in
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.82 if style.show_title else 0.90])
    ax.set_xlim(-0.15, 4.15)
    ax.set_ylim(-0.02, 1.08)
    ax.axis("off")
    transparent = style.background == "transparent"
    fig.patch.set_alpha(0.0 if transparent else 1.0)
    fig.patch.set_facecolor("none" if transparent else "white")
    pal = PALETTE if style.palette == "categorical" else MONO
    all_labels = []
    for s in range(4):
        for lab in _order(c.nodes[s], "fixed"):
            if lab not in all_labels:
                all_labels.append(lab)
    color = {lab: pal[i % len(pal)] for i, lab in enumerate(all_labels)}
    if OTHER in color:
        color[OTHER] = "#9a9a9a"
    node_w = 0.09
    xs = [0, 1, 2, 3]
    for k, r in enumerate(ribbons):
        x0, x1 = xs[r["stage"]] + node_w / 2, xs[r["stage"] + 1] - node_w / 2
        p = _ribbon_path(x0, x1, r["src"][0], r["src"][1], r["tgt"][0], r["tgt"][1])
        ax.add_patch(PathPatch(p, facecolor=blend(color[r["source"]], style.ribbon_alpha), edgecolor="none", lw=0))
        if style.value_labels == "all":
            # the number of a ribbon, placed at a third or two thirds of the way so that neighbours stagger
            t = 0.36 if k % 2 == 0 else 0.64
            ys = (r["src"][0] + r["src"][1]) / 2
            yt = (r["tgt"][0] + r["tgt"][1]) / 2
            # position on the cubic curve between the two centres
            ym = (1 - t) ** 3 * ys + 3 * (1 - t) ** 2 * t * ys + 3 * (1 - t) * t ** 2 * yt + t ** 3 * yt
            ax.text(x0 + t * (x1 - x0), ym, str(r["count"]), ha="center", va="center", fontsize=fs * 0.72,
                    color="#222222", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
    for s in range(4):
        for lab, (bot, top) in cols[s].items():
            ax.add_patch(Rectangle((xs[s] - node_w / 2, bot), node_w, top - bot, facecolor=color[lab],
                                   edgecolor="#333333", lw=0.6))
            txt = lab if style.value_labels == "none" else f"{lab} ({c.nodes[s][lab]})"
            ax.text(xs[s] + node_w / 2 + 0.03, (bot + top) / 2, txt, ha="left", va="center", fontsize=fs,
                    color="#111111", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
        ax.text(xs[s], 1.045, STAGES[s], ha="center", va="bottom", fontsize=fs * 1.05, fontweight="bold",
                color="#111111")
    if style.show_title:
        fig.text(0.5, 0.955, title_for(c), ha="center", va="top", fontsize=fs * 1.1, color="#111111", wrap=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=transparent, facecolor="none" if transparent else "white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf)
    im.load()
    return im if transparent else im.convert("RGB")


def save_png(im: Image.Image, path: str, dpi: int = 150, meta: dict[str, str] | None = None) -> None:
    info = PngInfo()
    for k, v in (meta or {}).items():
        info.add_text(k, v)
    im.save(path, dpi=(dpi, dpi), pnginfo=info, optimize=True)


def render_to_array(c: Cohort, style: RenderStyle | None = None) -> np.ndarray:
    return np.asarray(render(c, style).convert("RGB"))
