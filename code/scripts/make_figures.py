#!/usr/bin/env python3
"""Figures of the article, drawn to the figure guidelines of Open Research Europe.

    python scripts/make_figures.py

Open Research Europe asks that graphs and diagrams use RGB colour, a white background, text of at least 8 pt in a
consistent typeface, a printed width of 75 mm (one column) to 150 mm (two columns), and are supplied as EPS files or
else as uncompressed TIFF files of at least 600 dpi. Every figure is therefore drawn at its printed size, 150 mm
wide, with all text at 8 pt in Liberation Sans (metrically identical to Arial), without transparency, and written as

  * <name>.eps and <name>.pdf   vector files, fonts embedded as TrueType,
  * <name>.tif                  RGB, uncompressed, 600 dpi at the printed size, sRGB profile, resolution in the tags,
  * <name>.png                  RGB, 300 dpi at the printed size, sRGB profile, resolution in the pHYs chunk.

Figure 2 contains two rendered diagrams of the benchmark, which are raster images. They are drawn from the flow
tables at 600 dpi of the printed size so that the TIFF keeps its full resolution.
figure_check.md, written next to the figures, lists what was measured in every file.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.font_manager import findfont  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image, ImageCms, TiffImagePlugin  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from sankeyqa.cohort import Cohort  # noqa: E402
from sankeyqa.render import render, RenderStyle  # noqa: E402

OUT = os.path.join(ROOT, "figures")
PX_FULL = 1770                       # 149.9 mm at 300 dpi
FONT, FS = "Liberation Sans", 8.0
AUTHORS = "N. D. Tantaroudas, A. J. McCracken, I. Karachalios, E. Papatheou"
CREATOR = f"matplotlib {matplotlib.__version__} (scripts/make_figures.py)"
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5fd6", "#d6457a", "#4fb3d9"]
KEY = json.load(open(os.path.join(ROOT, "results", "key_numbers.json")))


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "Arial", "DejaVu Sans"], "font.size": FS,
        "axes.titlesize": FS, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 4,
        "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def fsize(h_mm: float) -> tuple[float, float]:
    return (PX_FULL + 0.01) / 300, (round(h_mm / 25.4 * 300) + 0.01) / 300


def audit(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    fb = fig.bbox
    clipped = [t.get_text() for t, b in zip(texts, boxes)
               if b.x0 < fb.x0 - .5 or b.x1 > fb.x1 + .5 or b.y0 < fb.y0 - .5 or b.y1 > fb.y1 + .5]
    over = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            w, h = min(a.x1, b.x1) - max(a.x0, b.x0), min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0 and h > 0 and w * h > 0.12 * min(a.width * a.height, b.width * b.height):
                over.append((texts[i].get_text(), texts[j].get_text()))
    return {"name": name, "n_text": len(texts), "min_pt": min(t.get_fontsize() for t in texts),
            "max_pt": max(t.get_fontsize() for t in texts),
            "fonts": sorted({os.path.basename(findfont(t.get_fontproperties())) for t in texts}),
            "clipped": clipped, "overlaps": over}


def save(fig, name, title, check):
    os.makedirs(OUT, exist_ok=True)
    check.append(audit(fig, name))
    base = os.path.join(OUT, name)
    fig.savefig(base + ".pdf", metadata={"Title": title, "Author": AUTHORS, "Creator": CREATOR})
    fig.savefig(base + ".eps", metadata={"Creator": CREATOR})
    for dpi, ext in ((300, "png"), (600, "tif")):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
        if ext == "png":
            info = PngInfo()
            for k, v in (("Title", title), ("Author", AUTHORS), ("Software", CREATOR)):
                info.add_text(k, v)
            rgb.save(base + ".png", dpi=(dpi, dpi), icc_profile=SRGB, pnginfo=info, optimize=True)
        else:
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            tags[270], tags[315], tags[305] = title, AUTHORS, CREATOR
            rgb.save(base + ".tif", dpi=(dpi, dpi), icc_profile=SRGB, compression=None, tiffinfo=tags)
    plt.close(fig)
    print("wrote", base + ".{eps,pdf,tif,png}")


def write_check(check):
    rows = []
    for c in check:
        base = os.path.join(OUT, c["name"])
        png, tif = Image.open(base + ".png"), Image.open(base + ".tif")
        eps = open(base + ".eps", "rb").read().decode("latin-1")
        pdf = open(base + ".pdf", "rb").read()
        bb = [float(v) for v in re.search(r"%%BoundingBox:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", eps).groups()]
        comp = {1: "none"}.get(tif.tag_v2.get(259), str(tif.tag_v2.get(259)))
        pd_, td = float(png.info["dpi"][0]), float(tif.info["dpi"][0])
        rows.append([c["name"], f"{25.4 * png.width / pd_:.1f} × {25.4 * png.height / pd_:.1f}",
                     f"{png.width} × {png.height} px, {pd_:.0f} dpi, {png.mode}, sRGB {'yes' if png.info.get('icc_profile') else 'no'}",
                     f"{tif.width} × {tif.height} px, {td:.0f} dpi, {tif.mode}, compression {comp}, sRGB {'yes' if tif.info.get('icc_profile') else 'no'}",
                     f"{25.4 * (bb[2] - bb[0]) / 72:.1f} × {25.4 * (bb[3] - bb[1]) / 72:.1f}",
                     "TrueType" if ("/FontType 42" in eps and b"/FontFile2" in pdf) else "check",
                     f"{c['n_text']} items, {c['min_pt']:g} to {c['max_pt']:g} pt, {', '.join(c['fonts'])}",
                     ", ".join(c["clipped"] + [f"{a} / {b}" for a, b in c["overlaps"]]) or "none"])
    head = ["Figure", "Printed size (mm)", "PNG", "TIFF", "EPS (mm)", "Fonts embedded", "Text",
            "Clipped or overlapping text"]
    lines = ["# Figure check", "", "Measured from the figure files by scripts/make_figures.py. Open Research Europe "
             "asks for RGB colour, a white background, text of at least 8 pt in a consistent typeface, a printed "
             "width of 75 to 150 mm, and EPS files or uncompressed TIFF files of at least 600 dpi. The PNG files "
             "(300 dpi) are those placed in the Word and LaTeX manuscripts.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    open(os.path.join(OUT, "figure_check.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    bad = [c["name"] for c in check if c["min_pt"] < 8 or c["clipped"] or c["overlaps"]]
    print("figure check:", "all figures pass" if not bad else "CHECK " + ", ".join(bad))


# ------------------------------------------------------------------------------------------------ Figure 1
def box(ax, x, y, w, h, text, fc="#f3f6fb", ec="#2a78d6", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=FS, fontweight="bold" if bold else None,
            linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2))


def fig_pipeline(check):
    fig = plt.figure(figsize=fsize(62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 62)
    ax.axis("off")
    box(ax, 2, 40, 27, 17, "Questionnaire\nrole, experience,\nindustry, age, target")
    box(ax, 36, 40, 27, 17, "Simulated cohort\nof profiles\n(no real CVs)")
    box(ax, 70, 40, 27, 17, "Flow table\ngroups and moves\nat 2, 5, 10 years", bold=False)
    box(ax, 104, 40, 44, 17, "Questions with exact answers\n11 types, computed\nfrom the flow table")
    arrow(ax, 29, 48.5, 36, 48.5)
    arrow(ax, 63, 48.5, 70, 48.5)
    arrow(ax, 97, 48.5, 104, 48.5)
    box(ax, 70, 12, 27, 17, "Career map picture\n12 drawing styles", fc="#fdf1ea", ec="#eb6834")
    arrow(ax, 83.5, 40, 83.5, 29)
    box(ax, 2, 3, 58, 26, "Picture readers\nSmolVLM 256M and 500M\nMatCha-ChartQA, BLIP-VQA\nBLIP fine-tuned on 20 to 400 maps",
        fc="#fdf1ea", ec="#eb6834")
    arrow(ax, 70, 20.5, 60, 20.5)
    box(ax, 104, 3, 44, 26, "Table readers\nQwen2.5-0.5B on the table text\nexact query on the table",
        fc="#eaf7f1", ec="#1baf7a")
    ax.add_patch(FancyArrowPatch((90, 40), (115, 29), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2,
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(31, 33.5, "every reply is scored against the exact answer", ha="center", va="center", fontsize=FS,
            color=INK2)
    save(fig, "fig1_pipeline", "SankeyQA pipeline", check)


# ------------------------------------------------------------------------------------------------ Figure 2
def fig_examples(check):
    sp = json.load(open(os.path.join(ROOT, "data", "splits.json")))
    tabs = {c: json.load(open(os.path.join(ROOT, "data", "tables", f"{c}.json"))) for c in sp["test"][:80]}
    role = next(c for c, t in tabs.items() if t["config"]["map_type"] == "role" and t["config"]["max_nodes"] >= 6)
    ind = next(c for c, t in tabs.items() if t["config"]["map_type"] == "industry")
    fig = plt.figure(figsize=fsize(178))
    for k, (cid, lab) in enumerate(((role, "A"), (ind, "B"))):
        c = Cohort.from_table(tabs[cid])
        im = render(c, RenderStyle(long_side_px=3540))
        ax = fig.add_axes([0.0, 0.505 - 0.5 * k, 1.0, 0.48])
        ax.imshow(np.asarray(im), interpolation="lanczos")
        ax.axis("off")
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, fontsize=FS, fontweight="bold", va="top")
    save(fig, "fig2_examples", "Two career maps of the benchmark", check)
    # single role map for the shorter preprint (raster only, 300 dpi at 150 mm)
    im = render(Cohort.from_table(tabs[role]), RenderStyle(long_side_px=PX_FULL))
    im.save(os.path.join(OUT, "fig2_role_example.png"), dpi=(300, 300), icc_profile=SRGB)
    return role, ind


# ------------------------------------------------------------------------------------------------ Figure 3
NICE = {"exact": "Exact query on table", "table_qwen2.5-0.5b": "Qwen2.5-0.5B, table",
        "smolvlm-256m": "SmolVLM-256M, picture", "smolvlm-256m-tiled": "SmolVLM-256M tiled, picture",
        "smolvlm-500m": "SmolVLM-500M, picture", "matcha-chartqa": "MatCha-ChartQA, picture",
        "blip-vqa": "BLIP-VQA, picture", "prior": "Guess without the map",
        "blip-ft_random400": "BLIP-VQA fine-tuned on 400 maps"}
QT = ["top_node", "max_outflow", "link_compare", "greedy_path", "flow_exists", "trend", "node_count", "node_value",
      "share_pct", "link_value", "total"]
QTN = {"top_node": "Largest group", "max_outflow": "Main destination", "link_compare": "Larger of two flows",
       "greedy_path": "Follow main moves", "flow_exists": "Flow exists", "trend": "Rise or fall",
       "node_count": "Number of groups", "node_value": "Group size", "share_pct": "Group share",
       "link_value": "Flow size", "total": "Total profiles"}


SHORT = {"exact": "Exact\nquery", "table_qwen2.5-0.5b": "Qwen2.5\ntable", "smolvlm-500m": "SmolVLM\n500M",
         "smolvlm-256m": "SmolVLM\n256M", "matcha-chartqa": "MatCha\nChartQA", "blip-vqa": "BLIP\nVQA",
         "prior": "Guess\nno map", "blip-ft_random400": "BLIP\n400 maps"}


def fig_by_qtype(check, readers):
    def acc(r, q):
        if r.startswith("blip-ft_"):
            return KEY[f"ft_{r[len('blip-ft_'):]}"][q]
        return KEY.get(f"acc_{r}_{q}") or 0.0
    M = np.array([[acc(r, q) for r in readers] for q in QT]) * 100
    fig = plt.figure(figsize=fsize(92))
    ax = fig.add_axes([0.20, 0.02, 0.78, 0.84])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(readers)))
    ax.set_xticklabels([SHORT[r] for r in readers])
    ax.set_yticks(range(len(QT)))
    ax.set_yticklabels([QTN[q] for q in QT])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=FS,
                    color="white" if M[i, j] > 60 else INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    save(fig, "fig3_accuracy_by_question", "Accuracy by question type and reader", check)


# ------------------------------------------------------------------------------------------------ Figure 4
STYLE_N = {"transparent": "Transparent background", "no_values": "No numbers", "all_values": "Numbers on ribbons",
           "small_font": "Small text", "large_font": "Large text", "mono": "Shades of blue", "px512": "512 pixels",
           "px768": "768 pixels", "px1536": "1536 pixels", "fixed_order": "Fixed node order", "no_title": "No title"}


def fig_robust(check, readers):
    styles = [s for s in STYLE_N if any(f"rob_{r}_{s}" in KEY for r in readers)]
    fig = plt.figure(figsize=fsize(80))
    ax = fig.add_axes([0.22, 0.13, 0.76, 0.80])
    off = np.linspace(-0.25, 0.25, len(readers))
    for k, r in enumerate(readers):
        ys, xs, lo, hi = [], [], [], []
        for i, s in enumerate(styles):
            v = KEY.get(f"rob_{r}_{s}")
            if v is None:
                continue
            ys.append(i + off[k])
            xs.append(100 * v[0])
            lo.append(100 * (v[0] - v[1]))
            hi.append(100 * (v[2] - v[0]))
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3.5, color=SLOTS[k], ecolor=SLOTS[k], elinewidth=0.8,
                    capsize=0, label=NICE[r])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(-20, 20)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels([STYLE_N[s] for s in styles])
    ax.invert_yaxis()
    ax.set_xlabel("Change in correct answers against the default drawing (percentage points)")
    ax.legend(loc="lower right", ncol=1, frameon=True, facecolor="white", edgecolor="white")
    save(fig, "fig5_robustness", "Effect of drawing style on accuracy", check)


# ------------------------------------------------------------------------------------------------ Figure 5
def fig_finetune(check):
    ft = pd.read_csv(os.path.join(ROOT, "results", "tables", "finetune.csv"))
    fig = plt.figure(figsize=fsize(62))
    ax = fig.add_axes([0.09, 0.17, 0.55, 0.78])
    base = 100 * KEY["acc_blip-vqa"]
    ax.axhline(base, color=INK2, lw=0.8, ls="--")
    ax.text(390, base + 1.5, "BLIP-VQA before fine-tuning", ha="right", va="bottom", fontsize=FS, color=INK2)
    g = 100 * KEY["acc_prior"]
    ax.axhline(g, color=INK2, lw=0.8, ls=":")
    ax.text(390, g - 1.5, "Guess without the map", ha="right", va="top", fontsize=FS, color=INK2)
    for sel, col, mk in (("random", SLOTS[0], "o"), ("kmeans", SLOTS[1], "D")):
        g = ft[ft["Selection"] == sel]
        x = g["Diagrams"].astype(float).to_numpy()
        y = g["Correct (%)"].astype(float).to_numpy()
        lo = np.array([float(s.split(" to ")[0]) for s in g["95% CI"]])
        hi = np.array([float(s.split(" to ")[1]) for s in g["95% CI"]])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=mk, ms=4, color=col, elinewidth=0.8, capsize=0,
                    label="Random diagrams" if sel == "random" else "Diverse diagrams (k-means)")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([20, 50, 100, 200, 400])
    ax.set_xticklabels(["20", "50", "100", "200", "400"])
    ax.set_ylim(0, 65)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Training diagrams (same number of training steps)")
    ax.set_ylabel("Correct answers (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0))
    save(fig, "fig6_finetuning", "Fine-tuning BLIP on few or many diagrams", check)


# ------------------------------------------------------------------------------------------------ Figure 6
def fig_speed(check, readers):
    fig = plt.figure(figsize=fsize(70))
    ax = fig.add_axes([0.09, 0.16, 0.55, 0.80])
    for k, r in enumerate(readers):
        if r == "prior":
            continue
        x = KEY[f"sec_{r}"]
        y = 100 * KEY[f"acc_{r}"]
        ax.errorbar([x], [y], yerr=[[y - 100 * KEY[f"acc_{r}_lo"]], [100 * KEY[f"acc_{r}_hi"] - y]], fmt="o", ms=4,
                    color=SLOTS[k % len(SLOTS)], elinewidth=0.8, capsize=0, label=NICE[r])
    ax.set_xlim(-0.15, 3.5)
    ax.set_xlabel("Seconds per question on two processor cores")
    ax.set_ylabel("Correct answers (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "fig4_speed_accuracy", "Accuracy against answering time", check)


def main() -> int:
    style()
    check = []
    fig_pipeline(check)
    fig_examples(check)
    order = [r for r in ["exact", "table_qwen2.5-0.5b", "smolvlm-500m", "smolvlm-256m-tiled", "smolvlm-256m",
                         "matcha-chartqa", "blip-vqa", "prior"] if f"acc_{r}" in KEY]
    fig_by_qtype(check, order[:-1] + ["blip-ft_random400", "prior"] if "ft_random400" in KEY else order)
    rob = [r for r in ["smolvlm-256m", "blip-vqa", "blip-ft_random400"] if f"rob_{r}_default" in KEY]
    if rob:
        fig_robust(check, rob)
    if os.path.exists(os.path.join(ROOT, "results", "tables", "finetune.csv")):
        fig_finetune(check)
    fig_speed(check, order)
    write_check(check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
