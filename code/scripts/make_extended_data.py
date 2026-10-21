#!/usr/bin/env python3
"""Write the extended data of the article to extended_data/.

    python scripts/make_extended_data.py

ExtendedData1_question_templates.csv   the eleven question templates, answer types, scoring and counts
ExtendedData2_reader_inputs.md         the exact input of every reader, decoding, scoring and fine-tuning settings
ExtendedData3_drawing_styles.csv       the twelve drawing styles and their settings
ExtendedData4_simulation_settings.md   the questionnaire fields and the model that simulates the cohorts
ExtendedData5_full_results.xlsx        all result tables in full, with 95% intervals for every drawing style
ExtendedData6_style_examples.png       one test map drawn in all twelve styles
README.md                              description of the files

Templates are checked against the stored questions, and the drawing-style intervals are recomputed from the scored
replies with the same paired bootstrap as the article and checked against results/key_numbers.json.
"""
from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
from sankeyqa import cohort as C  # noqa: E402
from sankeyqa.questions import QTYPES  # noqa: E402

OUT = os.path.join(ROOT, "extended_data")
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------------------------- 1 question templates
ENT, ENTS = "{role | industry}", "{roles | industries}"
TEMPLATES = {
    "top_node": (f"Which {ENT} has the most profiles at the {{stage}} stage?", "5 years, 10 years"),
    "node_count": (f"How many different {ENTS} are shown at the {{stage}} stage, counting \"Other\" as one if it "
                   f"appears?", "2 years, 5 years, 10 years"),
    "node_value": ("How many profiles are {group} at the {stage} stage?", "2 years, 5 years, 10 years"),
    "share_pct": ("What percentage of all profiles are {group} at the {stage} stage? Answer with a number.",
                  "2 years, 5 years, 10 years"),
    "max_outflow": (f"Of the profiles that are {{group}} at the {{stage}} stage and are in a different {ENT} at the "
                    f"{{next stage}} stage, which {ENT} do most of them move to?", "Now, 2 years, 5 years"),
    "link_compare": ("Between the {stage} and {next stage} stages, which is larger, the flow from {group} to "
                     "{destination A} or the flow from {group} to {destination B}? Answer with the name of the "
                     "destination.", "Now, 2 years, 5 years"),
    "greedy_path": (f"Start from {{group}} at the {{stage}} stage. At every stage follow the largest flow that goes to "
                    f"a different {ENT}. Which {ENT} do you reach at the 10 years stage?", "Now, 2 years"),
    "flow_exists": ("Is there any flow from {group} at the {stage} stage to {group B} at the {next stage} stage? "
                    "Answer yes or no.", "Now, 2 years, 5 years"),
    "link_value": ("How many profiles move from {group} at the {stage} stage to {group B} at the {next stage} stage?",
                   "Now, 2 years, 5 years"),
    "trend": ("Does the number of {group} profiles increase, decrease or stay the same between the {stage} and "
              "{next stage} stages?", "2 years to 5 years, 5 years to 10 years"),
    "total": ("How many profiles does this career map include in total?", "whole map"),
}
ANSWER = {"label": "a name from the map", "integer": "a whole number", "number": "a number", "yesno": "yes or no",
          "trend": "increase, decrease or stay the same"}
SCORING = {"label": "correct when the first possible name in the reply is the right one",
           "yesno": "correct when the first yes or no in the reply is right",
           "trend": "correct when the first trend word in the reply is right",
           "integer": "correct within 5% of the exact value or within one unit, exact match also recorded",
           "number": "correct within 5% of the exact value or within one unit, exact match also recorded"}
SEE = {"labels": "names and sizes", "node_values": "numbers beside the names", "link_values": "numbers on the bands",
       "title": "the total in the title"}

qs = [json.loads(l) for l in open(os.path.join(DATA, "questions.jsonl"), encoding="utf-8")]
splits = json.load(open(os.path.join(DATA, "splits.json")))
test = set(splits["test"])
main_n = {r["Question type"]: r["n"] for r in csv.DictReader(open(os.path.join(RES, "tables", "main_by_qtype.csv")))}
# examples of Table 1 of the article (test split)
TABLE1 = {
    "top_node": "Which role has the most profiles at the 10 years stage?",
    "node_count": "How many different roles are shown at the 10 years stage, counting",
    "node_value": "How many profiles are Manager at the 2 years stage?",
    "share_pct": "What percentage of all profiles are Consultant at the 2 years stage?",
    "max_outflow": "Of the profiles that are Team Lead at the Now stage and are in a different role at the 2 years",
    "link_compare": "Between the Now and 2 years stages, which is larger, the flow from Team Lead to Specialist or "
                    "the flow from Team Lead to Other?",
    "greedy_path": "Start from Team Lead at the Now stage.",
    "flow_exists": "Is there any flow from Specialist at the 5 years stage to Other at the 10 years stage?",
    "link_value": "How many profiles move from Team Lead at the Now stage to Team Lead at the 2 years stage?",
    "trend": "Does the number of Team Lead profiles increase, decrease or stay the same between the 5 years and 10",
    "total": "How many profiles does this career map include in total?",
}


def template_regex(t: str) -> re.Pattern:
    parts = re.split(r"(\{[^}]*\})", t)
    rx = "".join(".+?" if p.startswith("{") else re.escape(p) for p in parts)
    return re.compile("^" + rx + "$")


rows1, example_cohort = [], None
for qt in QTYPES:
    tq = [q for q in qs if q["qtype"] == qt]
    rx = template_regex(TEMPLATES[qt][0])
    bad = [q["text"] for q in tq if not rx.match(q["text"])]
    assert not bad, (qt, bad[:2])
    ex = next(q for q in tq if q["cohort"] in test and q["text"].startswith(TABLE1[qt]))
    if qt == "top_node":
        example_cohort = ex["cohort"]
    name = A.QTYPE_NAMES[qt]
    rows1.append({
        "Question type": name, "Code name": qt, "Template": TEMPLATES[qt][0], "Stages asked": TEMPLATES[qt][1],
        "Answer": ANSWER[tq[0]["answer_type"]], "Scoring": SCORING[tq[0]["answer_type"]],
        "What the reader must see": SEE[tq[0]["needs"][-1]],
        "Questions in the benchmark": len(tq), "Questions in the training split": sum(q["cohort"] not in test for q in tq),
        "Questions in the test split": sum(q["cohort"] in test for q in tq),
        "Questions in the main evaluation": main_n[name],
        "Example (test split)": ex["text"], "Exact answer of the example": ex["answer"],
    })
assert sum(r["Questions in the benchmark"] for r in rows1) == len(qs) == 6560
assert sum(int(r["Questions in the main evaluation"]) for r in rows1) == 875
with open(os.path.join(OUT, "ExtendedData1_question_templates.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows1[0]))
    w.writeheader()
    w.writerows(rows1)

# ---------------------------------------------------------------------------------------------- 2 reader inputs
src = open(os.path.join(ROOT, "code", "sankeyqa", "models.py"), encoding="utf-8").read()
INSTRUCTION = next(ast.literal_eval(n.value) for n in ast.parse(src).body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "INSTRUCTION")
versions = json.load(open(os.path.join(RES, "model_versions.json")))
tab = json.load(open(os.path.join(DATA, "tables", f"{example_cohort}.json")))
ns: dict = {}
exec(compile(ast.Module(body=[n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)
                              and n.name == "table_text"], type_ignores=[]), "models.py", "exec"),
     {"Any": object}, ns)
table_example = ns["table_text"](tab)
ex_q = next(q for q in qs if q["cohort"] == example_cohort and q["qtype"] == "top_node")
rev = {k: v for k, v in versions.items() if "/" in k}
md2 = f"""# SankeyQA extended data 2, exact inputs of every reader

The career maps and all numbers are artificial. They were generated by a computer program only to demonstrate and
test the technology. This file documents exactly what every reader in the article received and how its reply was
scored, so that every number can be reproduced. All settings are taken from `code/sankeyqa/models.py`,
`code/scripts/run_eval.py`, `code/sankeyqa/scoring.py`, `code/scripts/analyze.py` and
`code/scripts/finetune_blip.py`.

## The instruction

SmolVLM and the table reader received the question after this instruction, word for word.

> {INSTRUCTION}

BLIP-VQA and MatCha-ChartQA were trained to answer questions directly and received the question alone.

## Readers

| Reader | Model on the Hugging Face hub (revision) | What it received | Image handling | Decoding |
|---|---|---|---|---|
| SmolVLM-256M and SmolVLM-500M | HuggingFaceTB/SmolVLM-256M-Instruct ({rev['HuggingFaceTB/SmolVLM-256M-Instruct'][:12]}), HuggingFaceTB/SmolVLM-500M-Instruct ({rev['HuggingFaceTB/SmolVLM-500M-Instruct'][:12]}) | one user message with the image and the text "instruction, new line, question", formatted with the model's chat template | one view, longest side resized to 512 pixels, no image splitting | greedy, at most 24 new tokens |
| MatCha-ChartQA | google/matcha-chartqa ({rev['google/matcha-chartqa'][:12]}) | the image and the question | at most 1,024 image patches | greedy, at most 24 new tokens |
| BLIP-VQA | Salesforce/blip-vqa-base ({rev['Salesforce/blip-vqa-base'][:12]}) | the image and the question | resized by the BLIP processor to 384 by 384 pixels | greedy, at most 12 new tokens |
| BLIP-VQA fine-tuned | the same model after fine-tuning (settings below) | the image and the question | as BLIP-VQA | as BLIP-VQA |
| Qwen2.5-0.5B reading the table | Qwen/Qwen2.5-0.5B-Instruct ({rev['Qwen/Qwen2.5-0.5B-Instruct'][:12]}) | a system message with the instruction and a user message with the flow table as text and the question (format below) | no image | greedy, at most 24 new tokens |
| Exact query on the table | none | the flow table | no image | the answer is recomputed from the table by a routine independent of the question generator |

All models ran on a processor in 32-bit precision. Images with a transparent background were flattened onto black,
which is what a reader that ignores transparency sees. Full revision hashes and library versions are in
`results/model_versions.json`.

Exact message formats, where <instruction> is the text above, <question> the question and <flow table> the table
as text (example below).

```
SmolVLM, user message      [image] + "<instruction>\\n<question>"
Qwen2.5-0.5B, system       "<instruction>"
Qwen2.5-0.5B, user         "<flow table>\\n\\nQuestion: <question>\\nShort answer:"
BLIP-VQA, MatCha-ChartQA   [image] + "<question>"
```

## Example of the flow table given to the table reader

For the test map whose question "{ex_q['text']}" appears in Table 1 of the article (exact answer {ex_q['answer']}),
the table reader received this text.

```
{table_example}
```

## Baselines

* Random choice among the possible answers. The expected accuracy of a uniform choice among the options of each
  question, one over the number of options for names, yes or no and trends, and zero for open numbers.
* Guess that never looks at the map. Name questions get the starting role or industry of the questionnaire when it is
  one of the options (otherwise the first option), yes or no and trend questions get the most frequent answer of
  their type in the training split, and number questions get the median answer of their type in the training split.

## Reading and scoring the replies

* Names, the option that occurs first in the reply counts (the longest option wins at the same position). Case,
  punctuation and spacing are ignored.
* Yes or no, the first "yes" or "no" in the reply.
* Rise or fall, the first word that means increase, decrease or stay the same (for example "increase", "grow",
  "rise", "decrease", "fall", "drop", "same", "unchanged").
* Numbers, the first number followed by %, "percent", "profiles" or "people", otherwise the first number. A number is
  correct when it is within 5% of the exact value, with a floor of one unit (one profile or one percentage point).
  Exact matches are recorded separately.
* A reply without a usable answer counts as wrong.
* Intervals are 95% percentile intervals from 2,000 bootstrap resamples of maps (all questions of a map are resampled
  together). Differences between readers or between drawing styles are paired on the same questions.

## Fine-tuning settings (BLIP-VQA)

* Training data, all questions of the chosen training diagrams in the default style.
* Diverse selection, k-means clustering (k = 20, 10 initialisations) of the mean image features of the 400 training
  diagrams after scaling each to unit length, keeping the diagram nearest to each cluster centre. Random selection,
  20, 100 or 400 diagrams drawn without replacement (20 diagrams with seeds 1, 2 and 3).
* The image encoder was frozen and its output computed once per diagram. The question encoder and the answer decoder
  were trained with AdamW (learning rate 0.00003, weight decay 0.01), 600 updates of 8 questions, a linear warm-up
  over 30 updates followed by a linear decrease to zero, questions truncated at 64 tokens.
* The chosen diagrams, settings and loss curve of every run are in `results/finetune_runs/*/train_info.json`.
"""
open(os.path.join(OUT, "ExtendedData2_reader_inputs.md"), "w", encoding="utf-8").write(md2)

# ---------------------------------------------------------------------------------------------- 3 drawing styles
styles = json.load(open(os.path.join(DATA, "styles.json")))
STYLE_DESC = {"default": "Default drawing", "transparent": "Transparent background", "no_values": "No numbers",
              "all_values": "Numbers also on the bands", "small_font": "Small text", "large_font": "Large text",
              "mono": "Shades of blue", "px512": "512 pixels wide", "px768": "768 pixels wide",
              "px1536": "1,536 pixels wide", "fixed_order": "Fixed order of the groups", "no_title": "No title"}
n_img = {s: len(os.listdir(os.path.join(DATA, "images", s))) for s in styles}
assert sum(n_img.values()) == 1260
rows3 = []
for s, v in styles.items():
    rows3.append({"Style": s, "Description": STYLE_DESC[s], "Background": v["background"],
                  "Text size (relative)": v["font_scale"],
                  "Numbers shown": {"none": "none", "nodes": "beside each group", "all": "beside each group and on "
                                    "every band"}[v["value_labels"]],
                  "Colours": {"categorical": "one colour per group", "mono": "shades of blue"}[v["palette"]],
                  "Longer side (pixels)": v["long_side_px"], "Title with the total": "yes" if v["show_title"] else "no",
                  "Order of the groups": {"size": "largest at the top", "fixed": "fixed ladder order"}[v["node_order"]],
                  "Band opacity": v["ribbon_alpha"], "Images": n_img[s],
                  "Maps drawn": "all 600 cohorts" if s == "default" else "the first 60 test cohorts",
                  "Used in": "main evaluation, fine-tuning and drawing-style test" if s == "default"
                  else "drawing-style test"})
with open(os.path.join(OUT, "ExtendedData3_drawing_styles.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows3[0]))
    w.writeheader()
    w.writerows(rows3)

# ---------------------------------------------------------------------------------------------- 4 simulation settings
rel_rows = "\n".join(f"| {k} | {', '.join(v)} |" for k, v in C.RELATED.items())
md4 = f"""# SankeyQA extended data 4, questionnaire fields and simulation of the career cohorts

The career cohorts are artificial. They were generated by the program below only to demonstrate and test the
technology, they contain no personal data and they say nothing about real careers. The settings are those of
`code/sankeyqa/cohort.py` and `code/scripts/make_dataset.py`.

## Questionnaire fields (one configuration per career map)

| Field | Values | How a configuration is drawn |
|---|---|---|
| Kind of map | roles or industries | roles with probability 0.6 |
| Starting role | {', '.join(C.LADDER[:5])} | probabilities 0.30, 0.20, 0.15, 0.25 and 0.10 |
| Years of experience | {', '.join(str(x) for x in C.EXPERIENCE)} | uniform |
| Age group | {', '.join(C.AGE_GROUPS)} | tied to experience, group number = (years + a whole number from -3 to 3) divided by 8, rounded down, limited to the four groups |
| Industry | {', '.join(C.INDUSTRIES)} | uniform |
| Target role | a ladder role above the starting role | uniform among the higher roles (Executive for a Director) |
| Number of matched profiles | 60, 90, 120, 180, 240, 320, 450, 600, 800, 1200 | uniform |
| Groups shown per stage | 4, 5, 6 or 7 | uniform, the remaining groups are merged into Other |

Stages, {', '.join(C.STAGES)} (steps of 2, 3 and 5 years). Roles, the ladder {', '.join(C.LADDER)} and the side roles
{', '.join(C.LATERAL)}. Every configuration is unique.

## Role maps

Every simulated person starts in the starting role and moves at each step with these probabilities.

* Experience factor e = 1 / (1 + years of experience / 12). Age factor 1.15, 1.0, 0.8 and 0.6 for the four age groups.
* Rate = 0.22 x (0.6 + 0.8 e) x age factor x u, with u uniform between 0.75 and 1.3 for each step.
* Probability of leaving the role during a step of t years, m = min(0.92, 1 - exp(-rate x t)).
* From a ladder role, stay 1 - m, move up 0.72 m, move down one level 0.10 m, move to a side role 0.18 m (Consultant
  0.65, Founder 0.35). Upward moves are spread over the higher levels with weight exp(-0.9 (levels jumped - 1)),
  multiplied by 1.6 up to the target role and 0.5 above it. At the top a move up counts as staying, at the bottom a
  move down counts as staying.
* From a side role, return to the ladder with probability b = 0.25 x a number uniform between 0.6 and 1.4, spread over
  the levels with weight exp(-0.6 x the number of levels between that level and Manager), otherwise stay.
* Every probability is multiplied by a number uniform between 0.85 and 1.15 and every row is scaled to sum to one.

## Industry maps

* The industries of a map are the starting industry, its related industries and random others, 4 to 7 in total.
* Probability of staying during a step of t years, max(0.5, exp(-mu x t)), with mu = 0.08 x a number uniform between
  0.6 and 1.5. Moves go to the other industries of the map with weight 3 for a related industry and 1 otherwise, each
  multiplied by a number uniform between 0.7 and 1.3.

| Industry | Related industries |
|---|---|
{rel_rows}

## From cohorts to maps and questions

* The counts of every stage and every band form the flow table. The largest groups of each stage are shown and the rest
  merged into Other. An automatic check confirms that the bands leaving and entering every group add up to its size.
* 600 configurations were drawn with seed 2026. The first 400 maps form the training split and the last 200 the test
  split, and the first 60 test maps were also drawn in the eleven other styles. Each cohort is simulated with a seed
  derived from its configuration, so the benchmark can be regenerated byte for byte with
  `python code/scripts/make_dataset.py --out data`.
* About one question of every type was generated for every map (templates in extended data 1), skipping templates that
  do not apply, for example when two groups tie.
"""
open(os.path.join(OUT, "ExtendedData4_simulation_settings.md"), "w", encoding="utf-8").write(md4)

# ---------------------------------------------------------------------------------------------- 5 full results
rob = A.load("robust")
rob["reader"] = rob["reader"].map(lambda r: "blip-ft_" + os.path.basename(os.path.normpath(r.split(":", 1)[1]))
                                  if r.startswith("blip-ft:") else r.replace(":", "_"))
key = json.load(open(os.path.join(RES, "key_numbers.json")))
rows5 = []
for rd in ["blip-ft_random400", "blip-vqa", "smolvlm-256m"]:
    g = rob[rob["reader"] == rd]
    d0 = g[g["style"] == "default"]
    acc0, lo0, hi0 = A.boot_ci(d0)
    rows5.append({"Reader": A.READER_NAMES[rd], "Style": "Default drawing", "Questions": len(d0),
                  "Correct (%)": round(100 * acc0, 1), "95% interval (%)": f"{100 * lo0:.1f} to {100 * hi0:.1f}",
                  "Change from the default (points)": "", "95% interval of the change (points)": ""})
    for st in styles:
        if st == "default":
            continue
        gs = g[g["style"] == st]
        d, lo, hi = A.boot_diff(gs, d0)
        kd = key[f"rob_{rd}_{st}"]
        assert abs(kd[0] - d) < 1e-12 and abs(kd[1] - lo) < 1e-12 and abs(kd[2] - hi) < 1e-12, (rd, st)
        acc, alo, ahi = A.boot_ci(gs)
        rows5.append({"Reader": A.READER_NAMES[rd], "Style": STYLE_DESC[st], "Questions": len(gs),
                      "Correct (%)": round(100 * acc, 1), "95% interval (%)": f"{100 * alo:.1f} to {100 * ahi:.1f}",
                      "Change from the default (points)": f"{100 * d:+.1f}",
                      "95% interval of the change (points)": f"{100 * lo:+.1f} to {100 * hi:+.1f}"})
styles_df = pd.DataFrame(rows5)
styles_df.to_csv(os.path.join(RES, "tables", "robustness_intervals.csv"), index=False)
A.write(styles_df, "robustness_intervals")

sheets = [
    ("Contents", None, ""),
    ("Overall", "main_overall", "Correct answers of every reader on the 875 test questions about 80 unseen maps "
     "(Table 3 of the article)."),
    ("Against table reader", "main_vs_table_reader", "Paired difference from Qwen2.5-0.5B reading the table "
     "(Table 4)."),
    ("By question type", "main_by_qtype", "Correct answers (%) by question type and reader (Table 5 and Figure 3)."),
    ("By kind of map", "main_by_map_type", "Correct answers (%) on role maps and industry maps (Table 6)."),
    ("Styles with intervals", "robustness_intervals", "Correct answers in every drawing style with 95% intervals, and "
     "the change from the default style with paired 95% intervals, 220 questions about 20 test maps per style "
     "(Table 7 and Figure 5)."),
    ("Styles by question type", "robustness_by_qtype", "Correct answers (%) by drawing style and question type."),
    ("Fine-tuning", "finetune", "BLIP-VQA fine-tuned on different training diagrams (Table 8 and Figure 6)."),
]
with pd.ExcelWriter(os.path.join(OUT, "ExtendedData5_full_results.xlsx"), engine="openpyxl") as xw:
    contents = [{"Sheet": s, "Content": d} for s, n, d in sheets if n]
    contents += [{"Sheet": "", "Content": "Intervals are 95% percentile intervals from 2,000 bootstrap resamples of "
                  "maps. All career data are artificial, generated only to demonstrate and test the technology. "
                  "Licence CC BY 4.0. The same tables are in results/tables/ as CSV files."}]
    pd.DataFrame(contents).to_excel(xw, sheet_name="Contents", index=False)
    for s, n, d in sheets[1:]:
        pd.read_csv(os.path.join(RES, "tables", f"{n}.csv")).to_excel(xw, sheet_name=s, index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = min(80, max(10, width + 2))

# ---------------------------------------------------------------------------------------------- 6 style examples
cid = splits["robust"][0] if "robust" in splits else splits["test"][0]
font = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
fb = ImageFont.truetype(font, 34)
TW, TH, PAD, CAP = 1000, 760, 30, 60
order = list(styles)
cols, rws = 3, 4
sheet = Image.new("RGB", (cols * TW + (cols + 1) * PAD, rws * (TH + CAP) + (rws + 1) * PAD), "white")
dr = ImageDraw.Draw(sheet)
for k, st in enumerate(order):
    im = Image.open(os.path.join(DATA, "images", st, f"{cid}.png"))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        chk = Image.new("RGBA", im.size, (255, 255, 255, 255))
        cd = ImageDraw.Draw(chk)
        for y in range(0, im.size[1], 24):
            for x in range(0, im.size[0], 24):
                if (x // 24 + y // 24) % 2:
                    cd.rectangle([x, y, x + 23, y + 23], fill=(222, 222, 222, 255))
        im = Image.alpha_composite(chk, im)
    im = im.convert("RGB")
    native = im.size
    im.thumbnail((TW, TH), Image.LANCZOS)
    x0 = PAD + (k % cols) * (TW + PAD)
    y0 = PAD + (k // cols) * (TH + CAP + PAD)
    sheet.paste(im, (x0 + (TW - im.size[0]) // 2, y0 + (TH - im.size[1]) // 2))
    dr.rectangle([x0, y0, x0 + TW - 1, y0 + TH - 1], outline=(195, 194, 183), width=2)
    dr.text((x0 + TW // 2, y0 + TH + 12), f"{STYLE_DESC[st]} ({native[0]} x {native[1]} pixels)", fill=(11, 11, 11),
            font=fb, anchor="mt")
sheet.save(os.path.join(OUT, "ExtendedData6_style_examples.png"), dpi=(300, 300), optimize=True)

# ---------------------------------------------------------------------------------------------- README
readme = f"""# SankeyQA, extended data

Extended data of the article "Reading the picture or reading the data? An open benchmark of career-map Sankey
diagrams for small vision-language models" (Open Research Europe). All career data are artificial. They were
generated by a computer program only to demonstrate and test the technology, contain no personal data and say
nothing about real careers. Licence CC BY 4.0. Regenerate with `python code/scripts/make_extended_data.py`.

| File | Content |
|---|---|
| ExtendedData1_question_templates.csv | The eleven question templates with their wording, the stages they ask about, the kind of answer, the scoring rule, what a reader must see in the map, the number of questions in the benchmark, the splits and the main evaluation, and one example from the test split with its exact answer |
| ExtendedData2_reader_inputs.md | The exact input of every reader (instruction, message format, image size, decoding), an example of the flow table given to the table reader, the two baselines, the rules that read and score the replies, and the fine-tuning settings |
| ExtendedData3_drawing_styles.csv | The twelve drawing styles, the setting that each changes, and the number of images drawn in each |
| ExtendedData4_simulation_settings.md | The questionnaire fields and value ranges, and the model that simulates the role and industry cohorts behind the maps |
| ExtendedData5_full_results.xlsx | All result tables of the article in full, including correct answers in every drawing style with 95% intervals and the change from the default style with paired 95% intervals |
| ExtendedData6_style_examples.png | One test map drawn in all twelve styles (300 dpi). In the transparent style the checkerboard marks the transparent background |

The test map shown in ExtendedData6 is cohort {cid}. Its flow table is `data/tables/{cid}.json` of the underlying data.
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)
print("extended data written to", OUT)
for fn in sorted(os.listdir(OUT)):
    print(f"  {fn}  {os.path.getsize(os.path.join(OUT, fn)):,} bytes")
