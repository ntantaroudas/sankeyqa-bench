#!/usr/bin/env python3
"""Build the Preprints.org version in the MDPI LaTeX template (journal option ``preprints``).

    python preprints/build_preprints.py [--zenodo 10.5281/zenodo.N] [--github URL]

Fills main.src.tex with the result tables, the robustness paragraph and the references (in order of first citation,
from paper/references.yaml, which paper/check_references.py verified), writes main.tex, compiles it with pdflatex
twice and checks the log for errors, undefined citations and references, and the text for colons and semicolons.
Writes preprints_upload.zip (main.tex, Definitions/, figures/) and supplementary_materials.zip (result tables).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results" / "tables"
ZENODO = "10.5281/zenodo.23449063"
GITHUB = "https://github.com/ntantaroudas/sankeyqa-bench"


def tex(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("–", "--").replace("×", "$\\times$"))


def rows(name):
    with open(RES / f"{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tab_overall() -> str:
    body = "\n".join(f"{tex(r['Reader'])} & {r['Correct (%)']} & {tex(r['95% CI']).replace(' to ', '--')} & "
                     f"{r['Unparsed (%)'] or '--'} & {r['Seconds per question']} \\\\" for r in rows("main_overall"))
    return ("\\begin{table}[H]\n\\small\n\\caption{Correct answers on the 875 test questions about 80 unseen maps. "
            "Interval, 95\\% bootstrap interval over maps. No answer, replies without a usable answer. Seconds per "
            "question on two processor cores.\\label{tab:overall}}\n\\isPreprints{\\centering}{}\n"
            "\\begin{tabularx}{\\textwidth}{X r r r r}\n\\toprule\nReader & Correct (\\%) & Interval & No answer (\\%) "
            "& Seconds \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}")


def tab_ft() -> str:
    lab = {"kmeans20": "20 diverse (k-means)", "random20_s1": "20 random, draw 1", "random20_s2": "20 random, draw 2",
           "random20_s3": "20 random, draw 3", "random100": "100 random", "random400": "400 random (all)"}
    by = {r["Run"]: r for r in rows("finetune")}
    body = "\n".join(f"{lab[k]} & {by[k]['Passes over the data']} & {by[k]['Correct (%)']} & "
                     f"{by[k]['95% CI'].replace(' to ', '--')} & {by[k]['Label questions (%)']} & "
                     f"{by[k]['Number questions (%)']} \\\\" for k in lab if k in by)
    return ("\\begin{table}[H]\n\\small\n\\caption{BLIP-VQA fine-tuned for 600 updates of 8 questions on different "
            "training maps, correct answers on the 875 test questions. Before fine-tuning BLIP-VQA answered 12.7\\% "
            "correctly.\\label{tab:ft}}\n\\isPreprints{\\centering}{}\n\\begin{tabularx}{\\textwidth}{X r r r r r}\n"
            "\\toprule\nTraining maps & Passes & Correct (\\%) & Interval & Name questions (\\%) & Number questions "
            "(\\%) \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}")


def bib(order: list[str], refs: dict) -> str:
    out = ["\\begin{thebibliography}{999}"]
    for k in order:
        r = refs[k]
        link = f"https://doi.org/{r['doi']}" if r.get("doi") else r["url"]
        out.append(f"\\bibitem{{{k}}}\n{tex(r['text'])}. \\href{{{link}}}{{{tex(link.replace('https://', ''))}}}")
    out.append("\\end{thebibliography}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", default=ZENODO)
    ap.add_argument("--github", default=GITHUB)
    a = ap.parse_args()
    refs = yaml.safe_load((ROOT / "paper" / "references.yaml").read_text(encoding="utf-8"))
    for r in refs.values():
        if isinstance(r.get("doi"), str):
            r["doi"] = r["doi"].replace("{{ZENODO_DOI}}", a.zenodo)
    src = (HERE / "main.src.tex").read_text(encoding="utf-8")
    rob = (HERE / "robustness_short.tex").read_text(encoding="utf-8").strip()
    src = src.replace("{{TAB_OVERALL}}", tab_overall()).replace("{{TAB_FT}}", tab_ft()).replace("{{ROBUST_SHORT}}", rob)
    src = src.replace("{{GITHUB_SHORT}}", a.github)
    order = []
    for m in re.finditer(r"\\cite\{([^}]+)\}", src):
        for k in m.group(1).split(","):
            k = k.strip()
            if k not in order:
                order.append(k)
    unknown = [k for k in order if k not in refs]
    src = src.replace("{{BIB}}", bib([k for k in order if k in refs], refs))
    (HERE / "main.tex").write_text(src, encoding="utf-8")
    # figures: PDF for line art, PNG for the rendered map
    (HERE / "figures").mkdir(exist_ok=True)
    for f in ("fig1_pipeline.pdf", "fig3_accuracy_by_question.pdf", "fig2_role_example.png"):
        shutil.copy(ROOT / "figures" / f, HERE / "figures" / f)
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=HERE,
                           capture_output=True, text=True)
    log = (HERE / "main.log").read_text(encoding="latin-1")
    ok = r.returncode == 0
    undef = re.findall(r"(Citation|Reference) `([^']+)' .*undefined", log)
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    # colons and semicolons in the text (commands, URLs, DOIs and the reference list excepted)
    body = src[src.index("\\Title"):src.index("\\reftitle")]
    body = re.sub(r"\\(url|href|ref|label|cite)\{[^}]*\}", " ", body)
    body = re.sub(r"\\corres\{[^}]*\}", " ", body)   # the template's own "Correspondence:" label
    body = re.sub(r"%.*", "", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    punct = [ln.strip()[:100] for ln in body.split("\n") if (":" in ln or ";" in ln)]
    print(f"pdflatex {'OK' if ok else 'FAILED'}, pages {pages.group(1) if pages else '?'}, undefined {undef or 'none'}, "
          f"unknown references {unknown or 'none'}, references {len(order)}")
    print(f"colons or semicolons in the text: {punct or 'none'}")
    if "XXXXXXX" in a.zenodo:
        print("NOTE the Zenodo DOI is still a placeholder, rebuild with --zenodo 10.5281/zenodo.<number>")
    with zipfile.ZipFile(HERE / "preprints_upload.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "main.tex", "main.tex")
        for p in sorted((HERE / "Definitions").iterdir()):
            z.write(p, f"Definitions/{p.name}")
        for p in sorted((HERE / "figures").iterdir()):
            z.write(p, f"figures/{p.name}")
    with zipfile.ZipFile(HERE / "supplementary_materials.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RES.iterdir()):
            z.write(p, f"result_tables/{p.name}")
        z.write(ROOT / "results" / "key_numbers.json", "key_numbers.json")
        z.write(ROOT / "results" / "model_versions.json", "model_versions.json")
        z.write(ROOT / "data" / "stats.json", "benchmark_stats.json")
    return 0 if ok and not undef and not unknown and not punct else 1


if __name__ == "__main__":
    sys.exit(main())
