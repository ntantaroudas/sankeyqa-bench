#!/usr/bin/env python3
"""Summarise the scored replies into the tables of the article.

    python scripts/analyze.py

Reads results/raw/<set>/*.jsonl and writes results/tables/*.csv and *.md and results/key_numbers.json.
Confidence intervals are 95% percentile intervals from 2,000 bootstrap resamples of cohorts (all questions of a
cohort are resampled together, because they share one diagram).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
from sankeyqa.questions import QTYPES  # noqa: E402
from sankeyqa.scoring import chance_level, score  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
TAB = os.path.join(ROOT, "results", "tables")
DATA = os.path.join(ROOT, "data")
B = 2000

READER_NAMES = {
    "exact": "Exact query on the table",
    "table_qwen2.5-0.5b": "Qwen2.5-0.5B reading the table",
    "table_qwen2.5-1.5b-int8": "Qwen2.5-1.5B (8-bit) reading the table",
    "smolvlm-256m": "SmolVLM-256M, one view",
    "smolvlm-256m-tiled": "SmolVLM-256M, tiled views",
    "smolvlm-500m": "SmolVLM-500M, one view",
    "matcha-chartqa": "MatCha-ChartQA",
    "blip-vqa": "BLIP-VQA (not fine-tuned)",
    "blip-ft_random400": "BLIP-VQA fine-tuned on 400 maps",
    "prior": "Training-set guess (no diagram)",
    "chance": "Random choice among options",
}
QTYPE_NAMES = {
    "top_node": "Largest group at a stage", "node_count": "Number of groups at a stage",
    "node_value": "Size of a group (printed)", "share_pct": "Share of a group (percent)",
    "max_outflow": "Main destination of movers", "link_compare": "Larger of two flows",
    "greedy_path": "Follow the main moves (3 steps or 2)", "flow_exists": "Does a flow exist",
    "link_value": "Size of a flow", "trend": "Rise or fall of a group", "total": "Total number of profiles",
}
NEEDS_NAMES = {"labels": "names only", "node_values": "numbers beside the names", "link_values": "numbers on ribbons",
               "title": "the title"}


def load(set_name: str) -> pd.DataFrame:
    rows = []
    for p in sorted(glob.glob(os.path.join(RAW, set_name, "*.jsonl"))):
        for l in open(p, encoding="utf-8"):
            rows.append(json.loads(l))
    return pd.DataFrame(rows)


def boot_ci(df: pd.DataFrame, col: str = "correct", seed: int = 0) -> tuple[float, float, float]:
    by = df.groupby("cohort")[col].agg(["sum", "count"])
    s, n = by["sum"].to_numpy(float), by["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    est = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> tuple[float, float, float]:
    """Paired difference of accuracy (a minus b) over the same questions, resampling cohorts."""
    m = a.merge(b, on=["qid", "cohort"], suffixes=("_a", "_b"))
    by = m.groupby("cohort").agg(sa=("correct_a", "sum"), sb=("correct_b", "sum"), n=("correct_a", "count"))
    sa, sb, n = by["sa"].to_numpy(float), by["sb"].to_numpy(float), by["n"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(n), size=(B, len(n)))
    est = (sa[idx].sum(1) - sb[idx].sum(1)) / n[idx].sum(1)
    return float((sa.sum() - sb.sum()) / n.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def write(df: pd.DataFrame, name: str) -> None:
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8").write(md(df))


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def prior_replies(questions: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """A guess that never looks at the diagram. Label questions get the starting label of the questionnaire when it
    is an option (the most frequent answer in the training split for top_node), other questions the most frequent
    training answer of their type, numbers the training median."""
    rows = []
    tables = {}
    for _, q in questions.iterrows():
        cid = q["cohort"]
        if cid not in tables:
            tables[cid] = json.load(open(os.path.join(DATA, "tables", f"{cid}.json")))
        cfg = tables[cid]["config"]
        start = cfg["start_role"] if cfg["map_type"] == "role" else cfg["industry"]
        tq = train[train["qtype"] == q["qtype"]]
        if q["answer_type"] == "label":
            guess = start if start in q["options"] else q["options"][0]
        elif q["answer_type"] in ("yesno", "trend"):
            guess = tq["answer"].astype(str).value_counts().index[0]
        else:
            guess = str(float(np.median(tq["answer"].astype(float))))
        sc = score(q.to_dict(), guess)
        rows.append({"qid": q["qid"], "cohort": cid, "qtype": q["qtype"], "style": "default", "reader": "prior",
                     "reply": guess, "answer": q["answer"], "seconds": 0.0, **sc})
    return pd.DataFrame(rows)


def main() -> int:
    qs = pd.DataFrame([json.loads(l) for l in open(os.path.join(DATA, "questions.jsonl"), encoding="utf-8")])
    sp = json.load(open(os.path.join(DATA, "splits.json")))
    key: dict = {}
    main_df = load("main")
    if main_df.empty:
        print("no results yet")
        return 0
    main_df["reader"] = main_df["reader"].str.replace(":", "_")
    ids80 = sp["test"][:80]
    mq = qs[qs["cohort"].isin(ids80)].copy()
    train = qs[qs["split"] == "train"]
    pri = prior_replies(mq, train)
    main_df = pd.concat([main_df, pri], ignore_index=True)
    key["n_main_questions"] = int(len(mq))
    key["n_main_cohorts"] = len(ids80)

    # Table, overall accuracy of every reader
    rows = []
    chance = float(np.mean([chance_level(q) for q in mq.to_dict("records")]))
    for rd, g in main_df.groupby("reader"):
        if len(g) < len(mq):
            print(f"note: {rd} has {len(g)} of {len(mq)} replies, left out of the main table")
            continue
        acc, lo, hi = boot_ci(g)
        ex = g["exact"].mean()
        rows.append({"Reader": READER_NAMES.get(rd, rd), "key": rd, "Questions": len(g),
                     "Correct (%)": pct(acc), "95% CI": f"{pct(lo)} to {pct(hi)}", "Exact (%)": pct(ex),
                     "Unparsed (%)": pct(1 - g["parsed"].mean()),
                     "Seconds per question": f"{g['seconds'].mean():.2f}" if rd != "prior" else "0"})
        key[f"acc_{rd}"] = acc
        key[f"acc_{rd}_lo"], key[f"acc_{rd}_hi"] = lo, hi
        key[f"sec_{rd}"] = float(g["seconds"].mean())
    rows.append({"Reader": READER_NAMES["chance"], "key": "chance", "Questions": len(mq), "Correct (%)": pct(chance),
                 "95% CI": "", "Exact (%)": "", "Unparsed (%)": "", "Seconds per question": "0"})
    key["acc_chance"] = chance
    overall = pd.DataFrame(rows).sort_values("Correct (%)", key=lambda s: s.astype(float), ascending=False)
    write(overall.drop(columns=["key"]), "main_overall")
    readers = [r for r in overall["key"] if r != "chance"]

    # Table, accuracy by question type
    rows = []
    for qt in QTYPES:
        row = {"Question type": QTYPE_NAMES[qt], "n": int((mq["qtype"] == qt).sum()),
               "Needs": ", ".join(NEEDS_NAMES[n] for n in mq[mq["qtype"] == qt]["needs"].iloc[0]
                                  if n != "labels" or len(mq[mq["qtype"] == qt]["needs"].iloc[0]) == 1)}
        for rd in readers:
            g = main_df[(main_df["reader"] == rd) & (main_df["qtype"] == qt)]
            row[READER_NAMES.get(rd, rd)] = pct(g["correct"].mean()) if len(g) else ""
            key[f"acc_{rd}_{qt}"] = float(g["correct"].mean()) if len(g) else None
        cl = float(np.mean([chance_level(q) for q in mq[mq["qtype"] == qt].to_dict("records")]))
        row["Random choice"] = pct(cl)
        rows.append(row)
    write(pd.DataFrame(rows), "main_by_qtype")

    # Table, picture readers versus the table reader, paired differences
    rows = []
    base = main_df[main_df["reader"] == "table_qwen2.5-0.5b"]
    if len(base):
        for rd in readers:
            if rd in ("table_qwen2.5-0.5b",):
                continue
            g = main_df[main_df["reader"] == rd]
            d, lo, hi = boot_diff(g, base)
            rows.append({"Reader": READER_NAMES.get(rd, rd), "Difference (points)": f"{100 * d:+.1f}",
                         "95% CI": f"{100 * lo:+.1f} to {100 * hi:+.1f}"})
            key[f"diff_{rd}_vs_table05"] = (d, lo, hi)
        write(pd.DataFrame(rows), "main_vs_table_reader")

    # Table, by map type (roles or industries)
    rows = []
    mt = dict(zip(qs["qid"], qs["map_type"]))
    main_df["map_type"] = main_df["qid"].map(mt)
    for rd in readers:
        g = main_df[main_df["reader"] == rd]
        row = {"Reader": READER_NAMES.get(rd, rd)}
        for t, nm in (("role", "Role maps (%)"), ("industry", "Industry maps (%)")):
            row[nm] = pct(g[g["map_type"] == t]["correct"].mean())
        rows.append(row)
    write(pd.DataFrame(rows), "main_by_map_type")

    # Robustness
    rob = load("robust")
    if not rob.empty:
        rob["reader"] = rob["reader"].map(lambda r: "blip-ft_" + os.path.basename(os.path.normpath(r.split(":", 1)[1]))
                                          if r.startswith("blip-ft:") else r.replace(":", "_"))
        styles = json.load(open(os.path.join(DATA, "styles.json")))
        rows = []
        for rd, g in rob.groupby("reader"):
            d0 = g[g["style"] == "default"]
            row = {"Reader": READER_NAMES.get(rd, rd), "Default (%)": pct(d0["correct"].mean())}
            for st in styles:
                if st == "default":
                    continue
                gs = g[g["style"] == st]
                if not len(gs):
                    continue
                d, lo, hi = boot_diff(gs, d0)
                row[st] = f"{100 * d:+.1f}"
                key[f"rob_{rd}_{st}"] = (d, lo, hi, float(gs["correct"].mean()))
            key[f"rob_{rd}_default"] = float(d0["correct"].mean())
            rows.append(row)
        write(pd.DataFrame(rows), "robustness")
        # correct answers by style and question type for every reader of the style experiment
        rows = []
        for (rd, st), g in rob.groupby(["reader", "style"]):
            row = {"Reader": READER_NAMES.get(rd, rd), "Style": st, "All (%)": pct(g["correct"].mean())}
            for qt in QTYPES:
                row[QTYPE_NAMES[qt]] = pct(g[g["qtype"] == qt]["correct"].mean())
            rows.append(row)
        write(pd.DataFrame(rows), "robustness_by_qtype")
        key["n_robust_questions_per_style"] = int(len(rob[(rob["style"] == "default")].groupby("qid")))

    # Fine-tuning
    ft = load("finetune")
    if not ft.empty:
        rows = []
        for rd, g in ft.groupby("reader"):
            run = rd.split("blip-ft_")[-1].split("/")[-1] if "blip-ft" in rd else rd
            info_p = os.path.join(ROOT, "results", "finetune_runs", run, "train_info.json")
            info = json.load(open(info_p)) if os.path.exists(info_p) else {}
            acc, lo, hi = boot_ci(g)
            per = {qt: g[g["qtype"] == qt]["correct"].mean() for qt in QTYPES}
            rows.append({"Run": run, "Selection": info.get("selection", ""), "Diagrams": info.get("n_images", ""),
                         "Training questions": info.get("n_questions", ""),
                         "Passes over the data": f"{info.get('epochs', 0):.1f}",
                         "Correct (%)": pct(acc), "95% CI": f"{pct(lo)} to {pct(hi)}",
                         "Label questions (%)": pct(g[g["qtype"].isin(["top_node", "max_outflow", "link_compare",
                                                                        "greedy_path"])]["correct"].mean()),
                         "Number questions (%)": pct(g[g["qtype"].isin(["node_count", "node_value", "share_pct",
                                                                         "link_value", "total"])]["correct"].mean())})
            key[f"ft_{run}"] = {"acc": acc, "lo": lo, "hi": hi, **{k: float(v) for k, v in per.items()}}
        write(pd.DataFrame(rows).sort_values("Run"), "finetune")

    json.dump(key, open(os.path.join(ROOT, "results", "key_numbers.json"), "w"), indent=1, default=float)
    print(open(os.path.join(TAB, "main_overall.md")).read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
