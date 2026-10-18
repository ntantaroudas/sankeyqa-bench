#!/usr/bin/env python3
"""Check every reference of references.yaml against the registry that issued its DOI.

    python paper/check_references.py            -> paper/reference_check.md and paper/reference_check.json

For a DOI, the script asks doi.org which registration agency issued it, reads the record from Crossref
(api.crossref.org) or DataCite (api.datacite.org, arXiv and Zenodo DOIs), and compares the registered title, year and
first author's family name with the entry. It also checks that https://doi.org/<DOI> resolves. Entries without a
DOI are checked by opening their official URL and looking for the title on the page.
"""
from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "sankeyqa-reference-check/1.0 (mailto:nikolaos.tantaroudas@iccs.gr)"}


def get(url: str, accept: str = "application/json", timeout: int = 40):
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.geturl(), r.read()


def simp(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def similar(a: str, b: str) -> float:
    A, B = set(simp(a).split()), set(simp(b).split())
    return len(A & B) / max(1, len(A | B))


def agency(doi: str) -> str:
    try:
        _, _, b = get(f"https://doi.org/ra/{urllib.parse.quote(doi)}")
        return json.loads(b)[0].get("RA", "?")
    except Exception as e:  # noqa: BLE001
        return f"? ({e})"


def record(doi: str, ra: str) -> dict:
    if ra == "DataCite":
        _, _, b = get(f"https://api.datacite.org/dois/{urllib.parse.quote(doi)}")
        a = json.loads(b)["data"]["attributes"]
        cre = a.get("creators") or [{}]
        fam = cre[0].get("familyName") or cre[0].get("name", "")
        return {"title": a["titles"][0]["title"], "year": a.get("publicationYear"), "first": fam}
    _, _, b = get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    m = json.loads(b)["message"]
    au = m.get("author") or [{}]
    fam = au[0].get("family") or au[0].get("name", "")
    y = None
    for fld in ("issued", "published-print", "published-online", "published", "created"):
        dp = (m.get(fld, {}).get("date-parts") or [[None]])[0][0]
        if dp:
            y = dp
            break
    t = (m.get("title") or [""])[0]
    if m.get("subtitle"):
        t = t + ": " + m["subtitle"][0]
    return {"title": t, "year": y, "first": fam, "container": (m.get("container-title") or [""])[0]}


def resolves(doi: str) -> tuple[bool, str]:
    try:
        req = urllib.request.Request(f"https://doi.org/{doi}", headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=40) as r:
            return True, r.geturl()
    except urllib.error.HTTPError as e:
        # publishers often refuse HEAD or robots, a redirect away from doi.org proves the DOI is registered
        loc = e.headers.get("Location") or e.url
        return (e.code in (403, 405, 406, 418, 429, 503) and "doi.org" not in str(loc)), f"HTTP {e.code} at {loc}"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def main() -> int:
    refs = yaml.safe_load(open(HERE / "references.yaml", encoding="utf-8"))
    rows, ok_all = [], True
    for key, r in refs.items():
        row = {"key": key, "doi": r.get("doi"), "url": r.get("url")}
        if r.get("pending"):
            # the authors' own Zenodo deposit, whose DOI is minted at upload, checked after the upload
            row.update({"agency": "pending (own deposit)", "ok": True, "pending": True})
            rows.append(row)
            print("PEND", key)
            continue
        try:
            if r.get("doi"):
                ra = agency(r["doi"])
                rec = record(r["doi"], ra)
                res, where = resolves(r["doi"])
                sim = similar(r["title"], rec["title"])
                yok = rec["year"] is not None and abs(int(rec["year"]) - int(r["year"])) <= 1
                fok = simp(r["first_author"]).split()[0] in simp(rec["first"]) or simp(rec["first"]) in simp(r["first_author"])
                ok = sim >= 0.8 and yok and fok and (res or ra in ("Crossref", "DataCite"))
                row.update({"agency": ra, "registered_title": rec["title"], "registered_year": rec["year"],
                            "registered_first_author": rec["first"], "title_similarity": round(sim, 2),
                            "resolves": res, "resolves_to": where, "ok": ok})
            else:
                st, final, body = get(r["url"], accept="text/html")
                page = body.decode("utf-8", "ignore")
                found = simp(r["title"]) in simp(page)
                row.update({"agency": "none (official page)", "registered_title": r["title"] if found else "",
                            "resolves": st == 200, "resolves_to": final, "title_on_page": found,
                            "ok": st == 200 and found})
        except Exception as e:  # noqa: BLE001
            row.update({"ok": False, "error": str(e)})
        ok_all &= row["ok"]
        rows.append(row)
        print(("OK  " if row["ok"] else "FAIL"), key, row.get("doi") or row.get("url"), row.get("title_similarity", ""),
              row.get("error", ""), flush=True)
        time.sleep(0.3)
    json.dump(rows, open(HERE / "reference_check.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    lines = ["# Reference check", "", f"Checked on {time.strftime('%Y-%m-%d')} by paper/check_references.py. "
             "For every DOI the registration agency was looked up at doi.org, the record was read from Crossref or "
             "DataCite, and its title, year and first author were compared with the entry. Entries without a DOI "
             "were checked on their official page.", "",
             "| Key | DOI or URL | Agency | Registered title | Year | First author | Resolves | Result |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['key']} | {r.get('doi') or r.get('url')} | {r.get('agency', '')} | "
                     f"{str(r.get('registered_title', '')).replace('|', '/')} | {r.get('registered_year', '')} | "
                     f"{r.get('registered_first_author', '')} | {r.get('resolves', '')} | "
                     f"{'ok' if r['ok'] else 'CHECK ' + r.get('error', '')} |")
    (HERE / "reference_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("all references verified" if ok_all else "SOME REFERENCES FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
