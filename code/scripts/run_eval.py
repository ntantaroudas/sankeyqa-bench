#!/usr/bin/env python3
"""Ask every question of an evaluation set to one reader and write the scored replies.

    python scripts/run_eval.py --reader smolvlm-256m --set main
    python scripts/run_eval.py --reader smolvlm-256m-tiled --set main
    python scripts/run_eval.py --reader matcha-chartqa --set robust --styles all
    python scripts/run_eval.py --reader table:qwen2.5-1.5b --set main
    python scripts/run_eval.py --reader deplot+qwen2.5-1.5b --set main
    python scripts/run_eval.py --reader exact --set main

Readers
    <image model>             reads the PNG of the diagram (smolvlm-256m, smolvlm-500m, blip-vqa, matcha-chartqa,
                              blip-ft:<checkpoint dir>)
    smolvlm-*-tiled           SmolVLM with the image split into tiles (longest edge 1024 pixels) instead of one view
    table:<llm>               a language model that reads the flow table as text, never the picture
    deplot+<llm>              DePlot turns the picture into a table, the language model answers from that table
    exact                     the answer computed from the flow table (upper bound and consistency check)

Evaluation sets (fixed by data/splits.json)
    main     the first 80 test cohorts, default style, all questions
    robust   the first 20 cohorts of the robustness subset, in every style
Results are appended to results/raw/<set>/<reader>.jsonl, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sankeyqa import models as M  # noqa: E402
from sankeyqa.cohort import Cohort  # noqa: E402
from sankeyqa.questions import answer_from_table, Question  # noqa: E402
from sankeyqa.scoring import score  # noqa: E402

N_MAIN, N_ROBUST = 80, 20
DEPLOT_PROMPT = "Generate underlying data table of the figure below:"


def eval_set(data: str, which: str, styles: list[str]) -> list[tuple[dict, str]]:
    sp = json.load(open(os.path.join(data, "splits.json")))
    qs = [json.loads(l) for l in open(os.path.join(data, "questions.jsonl"), encoding="utf-8")]
    if which == "main":
        ids = sp["test"][:N_MAIN]
        st = ["default"]
    else:
        ids = sp["robust"][:N_ROBUST]
        st = styles
    keep = set(ids)
    order = {c: i for i, c in enumerate(ids)}
    sel = sorted([q for q in qs if q["cohort"] in keep], key=lambda q: (order[q["cohort"]], q["qid"]))
    return [(q, s) for s in st for q in sel]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--set", default="main", choices=["main", "robust"])
    ap.add_argument("--styles", default="all")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "results", "raw"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--outset", default=None, help="folder name of the results (default the set name)")
    a = ap.parse_args()
    styles = json.load(open(os.path.join(a.data, "styles.json")))
    st_list = list(styles) if a.styles == "all" else a.styles.split(",")
    items = eval_set(a.data, a.set, st_list)
    if a.limit:
        items = items[:a.limit]
    outset = a.outset or a.set
    os.makedirs(os.path.join(a.out, outset), exist_ok=True)
    fname = a.reader.replace(":", "_").replace("/", "_").replace("+", "_plus_")
    if a.reader.startswith("blip-ft:"):
        fname = "blip-ft_" + os.path.basename(os.path.normpath(a.reader.split(":", 1)[1]))
    out_path = os.path.join(a.out, outset, f"{fname}.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            done.add((r["qid"], r["style"]))
    todo = [(q, s) for q, s in items if (q["qid"], s) not in done]
    print(f"{a.reader} on {a.set}: {len(items)} items, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    r = a.reader
    model = None
    kind = "image"
    if r == "exact":
        kind = "exact"
    elif r.startswith("table:"):
        kind = "table"
        model = M.load(r.split(":", 1)[1])
    elif r.startswith("deplot+"):
        kind = "deplot"
        deplot = M.load("deplot")
        model = M.load(r.split("+", 1)[1])
        cache_dir = os.path.join(a.out, "..", "deplot_tables")
        os.makedirs(cache_dir, exist_ok=True)
    elif r.startswith("blip-ft:"):
        model = M.BlipVQA(path=r.split(":", 1)[1])
    elif r.endswith("-tiled"):
        from transformers import AutoProcessor
        model = M.load(r[: -len("-tiled")])
        model.proc = AutoProcessor.from_pretrained(model.repo, do_image_splitting=True)
        model.proc.image_processor.size = {"longest_edge": 1024}
    else:
        model = M.load(r)
        if r.startswith("smolvlm"):
            from transformers import AutoProcessor
            model.proc = AutoProcessor.from_pretrained(model.repo, do_image_splitting=False)
            model.proc.image_processor.size = {"longest_edge": 512}

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for k, (q, s) in enumerate(todo):
            img = os.path.join(a.data, "images", s, f"{q['cohort']}.png")
            extra = {}
            if kind == "exact":
                c = Cohort.from_table(json.load(open(os.path.join(a.data, "tables", f"{q['cohort']}.json"))))
                qq = Question(**{k2: q[k2] for k2 in ("qtype", "text", "answer", "answer_type", "options", "needs",
                                                      "meta")})
                t1 = time.time()
                reply = str(answer_from_table(c, qq))
                dt = time.time() - t1
            elif kind == "table":
                t = json.load(open(os.path.join(a.data, "tables", f"{q['cohort']}.json")))
                ctx = M.table_text(t)
                reply, dt = model.ask_text(ctx, q["text"])
                extra["context_chars"] = len(ctx)
            elif kind == "deplot":
                cp = os.path.join(cache_dir, f"{s}__{q['cohort']}.txt")
                dt_plot = 0.0
                if os.path.exists(cp):
                    tab = open(cp, encoding="utf-8").read()
                else:
                    tab, dt_plot = deplot.ask(img, DEPLOT_PROMPT)
                    tab = tab.replace("<0x0A>", "\n")
                    open(cp, "w", encoding="utf-8").write(tab)
                    open(cp + ".time", "w").write(str(dt_plot))
                if not dt_plot and os.path.exists(cp + ".time"):
                    dt_plot = float(open(cp + ".time").read())
                ctx = "Table extracted from a career-map chart:\n" + tab
                reply, dt = model.ask_text(ctx, q["text"])
                extra["deplot_seconds"] = dt_plot
                extra["context_chars"] = len(ctx)
            else:
                reply, dt = model.ask(img, q["text"])
            sc = score(q, reply)
            rec = {"qid": q["qid"], "cohort": q["cohort"], "qtype": q["qtype"], "style": s, "reader": a.reader,
                   "reply": reply, "answer": q["answer"], "seconds": round(dt, 4), **sc, **extra}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            if (k + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {k + 1}/{len(todo)}  {el / (k + 1):.2f} s/item, eta {el / (k + 1) * (len(todo) - k - 1) / 60:.0f} min",
                      flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
