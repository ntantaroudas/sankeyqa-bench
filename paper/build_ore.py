#!/usr/bin/env python3
"""Build the Open Research Europe manuscript in the official ORE Article Template (Word).

    python paper/build_ore.py

Source: paper/manuscript_ore.md (edit this). Output: paper/SankeyQA_ORE_manuscript.docx, written into the official
template by paper/ore_template/fill_template.py, and paper/manuscript_ore.expanded.md (citations numbered, reference
list filled). Markdown conventions are those of the authors' KyGround and INTERACT manuscripts. [@key] citations,
"::: table" and "::: figure <png>" blocks, ^x^ superscript, **bold**, *italic*, `code`, [text](url) links.

Checks, the build fails when one fails.
  * No colon or semicolon in the text of the article (links, DOIs and file names excepted), the authors' style.
  * Abstract at most 300 words with the headings Background, Methods, Results and Conclusions, no citations.
  * At most eight keywords.
  * Figure and table titles at most 15 words, figures and tables cited in the text in numerical order.
  * Every citation key is in references.yaml, every reference is cited, and references.yaml passed
    check_references.py (reference_check.json, all entries ok).
  * Every percentage and decimal number in Results, Discussion, Abstract and Conclusions occurs in the result files.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "manuscript_ore.md"
OUT_MD = HERE / "manuscript_ore.expanded.md"
OUT_JSON = HERE / "docx" / "manuscript_ore.json"
OUT_DOCX = HERE / "SankeyQA_ORE_manuscript.docx"
ZENODO_DOI = "10.5281/zenodo.23449063"
GITHUB = "https://github.com/ntantaroudas/sankeyqa-bench"
RES = ROOT / "results" / "tables"
CITE = re.compile(r"\[(@[A-Za-z0-9_]+(?:\s*,\s*@[A-Za-z0-9_]+)*)\]")
COMMENT = re.compile(r"<!--.*?-->", re.S)

INLINE = re.compile(r"\*\*(?P<b>.+?)\*\*|(?<![\w*])\*(?P<i>[^*\s][^*]*?)\*(?![\w*])|`(?P<c>[^`]+)`|\^(?P<s>[^^]+)\^|"
                    r"\[(?P<lt>[^\]]+)\]\((?P<lu>(?:https?://|mailto:)[^)]+)\)|"
                    r"(?P<url>https?://[^\s)<>\[\]]+[^\s.,)<>\[\]])")


def inline(text: str, base: dict | None = None) -> list[dict]:
    base = dict(base or {})
    runs, pos = [], 0
    for m in INLINE.finditer(text):
        if m.start() > pos:
            runs.append({**base, "t": text[pos:m.start()]})
        if m.group("b") is not None:
            runs += inline(m.group("b"), {**base, "b": True})
        elif m.group("i") is not None:
            runs += inline(m.group("i"), {**base, "i": True})
        elif m.group("c") is not None:
            runs.append({**base, "t": m.group("c"), "code": True})
        elif m.group("s") is not None:
            runs.append({**base, "t": m.group("s"), "sup": True})
        elif m.group("lt") is not None:
            runs.append({**base, "t": m.group("lt"), "link": m.group("lu")})
        elif m.group("url") is not None:
            runs.append({**base, "t": m.group("url"), "link": m.group("url")})
        pos = m.end()
    if pos < len(text):
        runs.append({**base, "t": text[pos:]})
    return [r for r in runs if r["t"]]


def parse_table(lines: list[str], head: str) -> dict:
    widths = None
    m = re.search(r"widths=([\d,]+)", head)
    if m:
        widths = [int(x) for x in m.group(1).split(",")]
    caption = inline(lines[0])
    rows, notes = [], []
    for ln in lines[1:]:
        if ln.startswith("note "):
            notes.append(inline(ln[5:].strip()))
        elif ln.startswith("|") and not re.match(r"^\|(\s*-+\s*\|)+\s*$", ln):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows.append([[inline(part) for part in c.split("<br>")] for c in cells])
    return {"type": "table", "caption": caption, "header": rows[0], "rows": rows[1:], "notes": notes, "widths": widths}


def to_blocks(md: str) -> list[dict]:
    blocks, lines, i = [], md.split("\n"), 0
    para: list[str] = []

    def flush():
        if para:
            blocks.append({"type": "p", "runs": inline(" ".join(para))})
            para.clear()
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("::: table"):
            flush()
            j = i + 1
            while lines[j].strip() != ":::":
                j += 1
            blocks.append(parse_table([x for x in lines[i + 1:j] if x.strip()], ln))
            i = j + 1
            continue
        if ln.startswith("::: figure"):
            flush()
            path = ln.split(None, 2)[2].strip()
            j = i + 1
            while lines[j].strip() != ":::":
                j += 1
            blocks.append({"type": "figure", "path": str(ROOT / path),
                           "legend": inline(" ".join(x.strip() for x in lines[i + 1:j] if x.strip()))})
            i = j + 1
            continue
        if not ln.strip():
            flush()
        elif ln.startswith("# "):
            flush()
            blocks.append({"type": "title", "runs": inline(ln[2:])})
        elif ln.startswith("## "):
            flush()
            blocks.append({"type": "h1", "runs": inline(ln[3:])})
        elif ln.startswith("### "):
            flush()
            blocks.append({"type": "h2", "runs": inline(ln[4:])})
        elif ln.startswith("- "):
            flush()
            blocks.append({"type": "bullet", "runs": inline(ln[2:])})
        elif ln.startswith("REF "):
            flush()
            n, ref = ln[4:].split(" ", 1)
            blocks.append({"type": "ref", "n": n, "runs": inline(ref)})
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return blocks


def render_numbers(nums: list[int]) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out) + "]"


def number_citations(text: str, refs: dict):
    order, unknown = [], []
    for m in CITE.finditer(text):
        for k in re.findall(r"@([A-Za-z0-9_]+)", m.group(1)):
            if k not in refs:
                unknown.append(k)
            elif k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}
    return CITE.sub(lambda m: render_numbers([num[k] for k in re.findall(r"@([A-Za-z0-9_]+)", m.group(1)) if k in num]),
                    text), order, unknown


def read_csv(name: str) -> list[dict]:
    with open(RES / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_table(caption: str, header: list[str], rows: list[list[str]], notes: list[str] = (), widths=None) -> str:
    w = f" widths={','.join(map(str, widths))}" if widths else ""
    out = [f"::: table{w}", caption, "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    out += [f"note {n}" for n in notes]
    out.append(":::")
    return "\n".join(out) + "\n"


QT_ORDER = ["top_node", "node_count", "node_value", "share_pct", "max_outflow", "link_compare", "greedy_path",
            "flow_exists", "link_value", "trend", "total"]
QT_NAME = {"top_node": "Largest group at a stage", "node_count": "Number of groups at a stage",
           "node_value": "Size of a group", "share_pct": "Share of a group", "max_outflow": "Main destination of movers",
           "link_compare": "Larger of two flows", "greedy_path": "Follow the main moves",
           "flow_exists": "Does a flow exist", "link_value": "Size of a flow", "trend": "Rise or fall of a group",
           "total": "Total number of profiles"}
ANS = {"label": "name", "integer": "number", "number": "number", "yesno": "yes or no",
       "trend": "increase, decrease or stay the same"}
STYLE_NAME = {"transparent": "Transparent background", "no_values": "Names without numbers",
              "all_values": "Numbers also on bands", "small_font": "Text 30% smaller", "large_font": "Text 40% larger",
              "mono": "Shades of one colour", "px512": "512 pixels wide", "px768": "768 pixels wide",
              "px1536": "1,536 pixels wide", "fixed_order": "Groups in a fixed order", "no_title": "No title"}


def _clean(t: str) -> str:
    return t.replace("? Answer with a number.", "?").replace(" Answer yes or no.", "").replace(
        " Answer with the name of the destination.", "").replace('"', "")


def tables() -> str:
    qs = [json.loads(l) for l in open(ROOT / "data" / "questions.jsonl", encoding="utf-8")]
    sp = json.load(open(ROOT / "data" / "splits.json"))
    main_ids = set(sp["test"][:80])
    out = []
    # Table 1
    rows = []
    for qt in QT_ORDER:
        ex = next(q for q in qs if q["qtype"] == qt and q["split"] == "test" and len(q["text"]) < 190)
        needs = [n for n in ex["needs"] if n != "labels"]
        need_txt = {"node_values": "numbers beside the names", "link_values": "numbers on the bands",
                    "title": "the total in the title"}
        rows.append([QT_NAME[qt], _clean(ex["text"]), ANS[ex["answer_type"]],
                     ", ".join(need_txt[n] for n in needs) if needs else "names and sizes",
                     str(sum(1 for q in qs if q["qtype"] == qt and q["cohort"] in main_ids))])
    out.append(md_table("**Table 1. The eleven question types with an example from the test split.** ",
                        ["Question type", "Example", "Answer", "What the reader must see", "Main set"], rows,
                        ["Main set, number of questions of each type in the main evaluation (80 test maps, 875 "
                         "questions). Names and sizes, the names of the groups and a comparison of bar or band sizes."],
                        widths=[16, 44, 12, 18, 8]))
    # Table 2
    mv = json.load(open(ROOT / "results" / "model_versions.json"))["parameters_millions"]
    rows = [["SmolVLM-256M-Instruct", f"{mv['SmolVLM-256M-Instruct']}", "picture, one view 512 pixels", "general vision-language model"],
            ["SmolVLM-500M-Instruct", f"{mv['SmolVLM-500M-Instruct']}", "picture, one view 512 pixels", "general vision-language model"],
            ["MatCha-ChartQA", f"{mv['matcha-chartqa']}", "picture, up to 1,024 patches", "chart reader trained on bar, line and pie charts"],
            ["BLIP-VQA base", f"{mv['blip-vqa-base']}", "picture, 384 × 384 pixels", "visual question answering, also fine-tuned here"],
            ["Qwen2.5-0.5B-Instruct", f"{mv['Qwen2.5-0.5B-Instruct']}", "flow table as text", "general language model"],
            ["Exact query", "none", "flow table as data", "look-up, the upper bound"],
            ["Guess without the map", "none", "nothing", "typical answer from the training split"],
            ["Random choice", "none", "nothing", "uniform among the possible answers"]]
    out.append(md_table("**Table 2. The readers compared in this study.**",
                        ["Reader", "Parameters (millions)", "Input", "Kind"], rows,
                        ["All models were run on a processor in 32-bit precision with greedy decoding. Versions in the "
                         "extended data."], widths=[22, 14, 28, 36]))
    # Table 3
    rows = [[r["Reader"], r["Correct (%)"], r["95% CI"], r["Exact (%)"], r["Unparsed (%)"], r["Seconds per question"]]
            for r in read_csv("main_overall")]
    out.append(md_table("**Table 3. Correct answers of every reader on the 875 test questions.**",
                        ["Reader", "Correct (%)", "95% interval", "Exact match (%)", "No usable answer (%)",
                         "Seconds per question"], rows,
                        ["Correct, exact for names, yes or no and trend answers, within 5% or one unit for numbers. "
                         "Exact match, the same number exactly. Intervals from 2,000 bootstrap resamples of maps. "
                         "Seconds on two processor cores."], widths=[30, 12, 16, 14, 14, 14]))
    # Table 4
    rows = [[r["Reader"], r["Difference (points)"], r["95% CI"]] for r in read_csv("main_vs_table_reader")]
    out.append(md_table("**Table 4. Difference from the table reader on the same questions.**",
                        ["Reader", "Difference (percentage points)", "95% interval"], rows,
                        ["Difference in the percentage of correct answers against Qwen2.5-0.5B reading the table, "
                         "paired on the 875 questions, intervals from 2,000 bootstrap resamples of maps."],
                        widths=[40, 30, 30]))
    # Table 5
    bq = read_csv("main_by_qtype")
    cols = [c for c in bq[0].keys() if c not in ("Question type", "n", "Needs")]
    short = {"Exact query on the table": "Exact", "Training-set guess (no diagram)": "Guess",
             "Qwen2.5-0.5B reading the table": "Qwen table", "SmolVLM-500M, one view": "SmolVLM 500M",
             "SmolVLM-256M, one view": "SmolVLM 256M", "MatCha-ChartQA": "MatCha", "BLIP-VQA (not fine-tuned)": "BLIP",
             "Random choice": "Random"}
    rows = [[r["Question type"], r["n"]] + [r[c] for c in cols] for r in bq]
    out.append(md_table("**Table 5. Correct answers (%) by question type and reader.**",
                        ["Question type", "n"] + [short.get(c, c) for c in cols], rows,
                        ["Exact, exact query on the table. Guess, guess without the map. Qwen table, Qwen2.5-0.5B "
                         "reading the table. Random, random choice among the possible answers (0 for open numbers)."],
                        widths=[22, 6] + [9] * len(cols)))
    # Table 6
    rows = [[r["Reader"], r["Role maps (%)"], r["Industry maps (%)"]] for r in read_csv("main_by_map_type")]
    out.append(md_table("**Table 6. Correct answers by kind of map.**", ["Reader", "Role maps (%)",
                        "Industry maps (%)"], rows, ["Role maps, 50 test maps. Industry maps, 30 test maps."],
                        widths=[50, 25, 25]))
    # Table 7
    rb = read_csv("robustness")
    rnames = [r["Reader"] for r in rb]
    rows = [["Default style"] + [r["Default (%)"] + " (correct %)" for r in rb]]
    for st in STYLE_NAME:
        if st in rb[0]:
            rows.append([STYLE_NAME[st]] + [r.get(st, "") for r in rb])
    out.append(md_table("**Table 7. Change in correct answers when the same maps are drawn in another style.**",
                        ["Drawing style"] + rnames, rows,
                        ["First row, percentage of correct answers in the default style. Other rows, change in "
                         "percentage points on the same 220 questions about 20 test maps. Intervals in the extended "
                         "data and in Figure 5."], widths=[28] + [72 // max(1, len(rnames))] * len(rnames)))
    # Table 8
    ft = read_csv("finetune")
    order = ["kmeans20", "random20_s1", "random20_s2", "random20_s3", "random100", "random400"]
    lab = {"kmeans20": "20 diverse (k-means)", "random20_s1": "20 random, draw 1", "random20_s2": "20 random, draw 2",
           "random20_s3": "20 random, draw 3", "random100": "100 random", "random400": "400 random (all)"}
    byrun = {r["Run"]: r for r in ft}
    rows = [[lab[k], byrun[k]["Training questions"], byrun[k]["Passes over the data"], byrun[k]["Correct (%)"],
             byrun[k]["95% CI"], byrun[k]["Label questions (%)"], byrun[k]["Number questions (%)"]]
            for k in order if k in byrun]
    out.append(md_table("**Table 8. BLIP-VQA fine-tuned on different training diagrams.**",
                        ["Training diagrams", "Training questions", "Passes over the data", "Correct (%)",
                         "95% interval", "Name questions (%)", "Number questions (%)"], rows,
                        ["Every run made 600 updates of 8 questions. Correct answers on the 875 test questions. Name "
                         "questions, largest group, main destination, larger of two flows and following the main moves. "
                         "Number questions, number of groups, size and share of a group, size of a flow and total. "
                         "BLIP-VQA before fine-tuning answered 12.7% correctly."],
                        widths=[22, 13, 13, 12, 14, 13, 13]))
    return "\n".join(out)


def robustness_text() -> str:
    p = HERE / "robustness_text.md"
    return p.read_text(encoding="utf-8").strip() if p.exists() else "ROBUSTNESS TEXT MISSING"


def ref_line(r: dict) -> str:
    link = f"https://doi.org/{r['doi']}" if r.get("doi") else r["url"]
    shown = link.replace("https://", "")
    return f"{r['text']}. [{shown}]({link})"


def section(text: str, heading: str) -> str:
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", text, re.S | re.M)
    return m.group(1) if m else ""


def prose_punctuation(text: str) -> list[str]:
    """Colons and semicolons in the article text, links, DOIs, file paths and the reference list excepted."""
    bad = []
    body = text.split("## References")[0] + "\n" + text.split("## Figure legends")[-1] if "## Figure legends" in text \
        else text.split("## References")[0]
    for ln in body.split("\n"):
        if ln.startswith(":::") or ln.startswith("REF "):
            continue
        # the labels that ORE prescribes for the software availability statement
        if re.match(r"^(Source code available from|Archived source code at time of publication|License):", ln):
            ln = ln.split(":", 1)[1]
        t = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", ln)
        t = re.sub(r"https?://\S+|mailto:\S+|`[^`]*`|\b\d{1,2}:\d{2}\b", " ", t)
        if ":" in t or ";" in t:
            bad.append(ln[:120])
    return bad


def known_numbers() -> set[str]:
    nums = set()
    for p in list((ROOT / "results").rglob("*.md")) + list((ROOT / "results").rglob("*.csv")) + \
            list((ROOT / "results").rglob("*.json")) + [ROOT / "data" / "stats.json"]:
        if p.is_file() and p.stat().st_size < 5_000_000:
            for n in re.findall(r"\d+\.\d+|\d+", p.read_text(encoding="utf-8", errors="ignore")):
                nums.add(n)
                if "." in n:
                    a, b = n.split(".")
                    nums.add(f"{a}.{b[:1]}")
                    try:
                        nums.add(f"{float(n):.1f}")
                        nums.add(f"{float(n):.2f}")
                        nums.add(f"{100 * float(n):.1f}")
                    except ValueError:
                        pass
    return nums


def check_numbers(text: str) -> list[str]:
    known = known_numbers()
    missing = []
    for h in ("Abstract", "Results", "Discussion", "Conclusions"):
        sec = section(text, h)
        sec = re.sub(r"\[[\d,–]+\]", " ", sec)
        for n in re.findall(r"(?<![\w.])(\d+\.\d+)(?![\w.])", sec):
            if n not in known:
                missing.append(f"{h} {n}")
    return sorted(set(missing))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", default=ZENODO_DOI, help="DOI of the Zenodo deposit, once minted")
    ap.add_argument("--github", default=GITHUB)
    a = ap.parse_args()
    refs = yaml.safe_load((HERE / "references.yaml").read_text(encoding="utf-8"))
    rc = json.load(open(HERE / "reference_check.json", encoding="utf-8"))
    src = COMMENT.sub("", SRC.read_text(encoding="utf-8")).strip() + "\n"
    src = src.replace("{{TABLES}}", tables()).replace("{{ROBUSTNESS}}", robustness_text())
    src = src.replace("{{ZENODO_DOI}}", a.zenodo).replace("{{GITHUB}}", a.github)
    for r in refs.values():
        if isinstance(r.get("doi"), str):
            r["doi"] = r["doi"].replace("{{ZENODO_DOI}}", a.zenodo)
    if "XXXXXXX" in a.zenodo:
        print("NOTE the Zenodo DOI is still a placeholder, rebuild with --zenodo 10.5281/zenodo.<number>")
    text, order, unknown = number_citations(src, refs)
    reflist = "\n".join(f"REF {i + 1}. {ref_line(refs[k])}" for i, k in enumerate(order))
    text = text.replace("## References\n", "## References\n\n" + reflist + "\n", 1)
    ok = True
    bad_refs = [r["key"] for r in rc if not r["ok"] and r["key"] in order]
    print(f"references: {len(order)} cited, unknown {sorted(set(unknown)) or 'none'}, "
          f"not cited {[k for k in refs if k not in order] or 'none'}, failed check {bad_refs or 'none'}")
    ok &= not unknown and not bad_refs and all(k in order for k in refs)
    ab = section(text, "Abstract")
    words = len(re.findall(r"\S+", re.sub(r"^#+\s*", "", ab, flags=re.M)))
    heads = re.findall(r"^### (\w+)", ab, re.M)
    ab_ok = words <= 300 and heads == ["Background", "Methods", "Results", "Conclusions"] and not re.search(r"\[\d", ab)
    ok &= ab_ok
    print(f"abstract: {words} words with headings {heads}: {'OK' if ab_ok else 'CHECK'}")
    kw = [k.strip() for k in section(text, "Keywords").strip().split(",") if k.strip()]
    ok &= len(kw) <= 8
    print(f"keywords: {len(kw)}: {'OK' if len(kw) <= 8 else 'TOO MANY'}")
    body = text.split("## Figure legends")[0]
    for kind in ("Table", "Figure"):
        caps = [int(n) for n in re.findall(rf"\*\*{kind} (\d+)\.", text)]
        first = []
        for m in re.finditer(rf"\b{kind}s? (\d+)(?:\s*(?:and|to|–)\s*(\d+))?", body):
            for g in m.groups():
                if g and int(g) not in first:
                    first.append(int(g))
        st = caps == list(range(1, len(caps) + 1)) and first == caps
        ok &= st
        print(f"{kind.lower()}s: captions {caps}, first cited {first}: {'OK' if st else 'CHECK'}")
    long_t = [m.group(1) for m in re.finditer(r"\*\*((?:Table|Figure) \d+)\.\s*([^*]*)\*\*", text)
              if len(m.group(2).split()) > 15]
    ok &= not long_t
    print(f"titles over 15 words: {long_t or 'none'}")
    punct = prose_punctuation(text)
    ok &= not punct
    print(f"colons or semicolons in the text: {punct or 'none'}")
    miss = check_numbers(text)
    ok &= not miss
    print(f"decimal numbers not found in the result files: {miss or 'none'}")
    OUT_MD.write_text(text, encoding="utf-8")
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps({"blocks": to_blocks(text)}, ensure_ascii=False, indent=1), encoding="utf-8")
    r = subprocess.run([sys.executable, str(HERE / "ore_template" / "fill_template.py"), str(OUT_JSON), str(OUT_DOCX)],
                       capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip()[-2000:])
    ok &= r.returncode == 0
    print("BUILD OK" if ok else "BUILD HAS PROBLEMS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
