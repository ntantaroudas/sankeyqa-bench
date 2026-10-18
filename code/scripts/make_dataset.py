#!/usr/bin/env python3
"""Generate the SankeyQA benchmark, cohorts, diagrams, questions and splits.

    python scripts/make_dataset.py --out data --n-cohorts 600 --seed 2026

Writes
    data/tables/<id>.json           the flow table of every cohort (the ground truth)
    data/images/default/<id>.png    every cohort drawn in the default style
    data/images/<style>/<id>.png    the robustness subset drawn in each alternative style
    data/questions.jsonl            one line per question (cohort id, split, question, answer, needs, metadata)
    data/splits.json                cohort ids of the train, test and robustness subsets
    data/styles.json                the rendering styles
    data/stats.json                 counts used in the article
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sankeyqa.cohort import sample_config, simulate, check_consistency  # noqa: E402
from sankeyqa.questions import make_questions, answer_from_table, QTYPES  # noqa: E402
from sankeyqa.render import render, RenderStyle, save_png, title_for  # noqa: E402

# The alternative rendering styles of the robustness study, each changes one setting of the default style.
STYLES = {
    "default": RenderStyle(name="default"),
    "transparent": RenderStyle(name="transparent", background="transparent"),
    "no_values": RenderStyle(name="no_values", value_labels="none"),
    "all_values": RenderStyle(name="all_values", value_labels="all"),
    "small_font": RenderStyle(name="small_font", font_scale=0.7),
    "large_font": RenderStyle(name="large_font", font_scale=1.4),
    "mono": RenderStyle(name="mono", palette="mono"),
    "px512": RenderStyle(name="px512", long_side_px=512),
    "px768": RenderStyle(name="px768", long_side_px=768),
    "px1536": RenderStyle(name="px1536", long_side_px=1536),
    "fixed_order": RenderStyle(name="fixed_order", node_order="fixed"),
    "no_title": RenderStyle(name="no_title", show_title=False),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-cohorts", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-robust", type=int, default=60, help="test cohorts drawn in every alternative style")
    ap.add_argument("--per-type", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    for st in STYLES:
        os.makedirs(os.path.join(a.out, "images", st), exist_ok=True)

    cohorts, seen = [], set()
    while len(cohorts) < a.n_cohorts:
        cfg = sample_config(rng)
        if cfg.key() in seen:
            continue
        seen.add(cfg.key())
        c = simulate(cfg)
        check_consistency(c)
        cohorts.append(c)
    ids = [c.config.key() for c in cohorts]
    # splits by cohort, the test cohorts are the last n_test, the robustness subset the first n_robust of them
    test_ids = ids[-a.n_test:]
    train_ids = ids[:-a.n_test]
    robust_ids = test_ids[:a.n_robust]
    split = {i: ("test" if i in set(test_ids) else "train") for i in ids}

    questions = []
    n_img = 0
    for k, c in enumerate(cohorts):
        cid = c.config.key()
        with open(os.path.join(a.out, "tables", f"{cid}.json"), "w", encoding="utf-8") as f:
            json.dump(c.table(), f, indent=1)
        styles = list(STYLES) if cid in set(robust_ids) else ["default"]
        for st in styles:
            im = render(c, STYLES[st])
            save_png(im, os.path.join(a.out, "images", st, f"{cid}.png"), dpi=150,
                     meta={"Title": title_for(c), "Software": "sankeyqa make_dataset.py", "Style": st})
            n_img += 1
        for q in make_questions(c, seed=a.seed * 1000 + k, per_type=a.per_type):
            assert answer_from_table(c, q) == q.answer
            d = q.as_dict()
            d.update({"cohort": cid, "split": split[cid], "map_type": c.config.map_type,
                      "qid": f"{cid}-{q.qtype}-{sum(1 for x in questions if x['cohort'] == cid and x['qtype'] == q.qtype)}"})
            questions.append(d)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(cohorts)} cohorts, {n_img} images, {len(questions)} questions, {time.time() - t0:.0f}s",
                  flush=True)
    with open(os.path.join(a.out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for d in questions:
            f.write(json.dumps(d) + "\n")
    json.dump({"train": train_ids, "test": test_ids, "robust": robust_ids}, open(os.path.join(a.out, "splits.json"), "w"),
              indent=1)
    json.dump({k: v.as_dict() for k, v in STYLES.items()}, open(os.path.join(a.out, "styles.json"), "w"), indent=1)

    by_type = collections.Counter(q["qtype"] for q in questions)
    by_split = collections.Counter(q["split"] for q in questions)
    by_map = collections.Counter(c.config.map_type for c in cohorts)
    nodes = [len(st) for c in cohorts for st in c.nodes[1:]]
    links = [len(c.links) for c in cohorts]
    stats = {
        "n_cohorts": len(cohorts), "n_train_cohorts": len(train_ids), "n_test_cohorts": len(test_ids),
        "n_robust_cohorts": len(robust_ids), "n_images": n_img, "n_styles": len(STYLES),
        "n_questions": len(questions), "questions_by_type": dict(by_type), "questions_by_split": dict(by_split),
        "cohorts_by_map_type": dict(by_map), "qtypes": QTYPES,
        "visible_nodes_per_stage_mean": float(np.mean(nodes)), "visible_nodes_per_stage_min": int(min(nodes)),
        "visible_nodes_per_stage_max": int(max(nodes)), "links_per_diagram_mean": float(np.mean(links)),
        "links_per_diagram_min": int(min(links)), "links_per_diagram_max": int(max(links)),
        "profiles_per_cohort_min": int(min(c.n_profiles for c in cohorts)),
        "profiles_per_cohort_max": int(max(c.n_profiles for c in cohorts)),
        "answer_distribution": {qt: dict(collections.Counter(str(q["answer"]) for q in questions if q["qtype"] == qt))
                                for qt in ("flow_exists", "trend")},
        "seed": a.seed, "seconds": time.time() - t0,
    }
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "answer_distribution"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
