"""Turn a model's free-text reply into an answer and score it against the exact answer.

Label questions are scored by the first candidate label that occurs in the reply (candidate labels are the
options of the question). Numbers are scored with the relaxed accuracy used by chart question answering
benchmarks, a reply counts as correct when it is within 5 percent of the exact value, with a floor of one unit
(one profile, or one percentage point). Exact matches are also recorded.
"""
from __future__ import annotations

import re
from typing import Any

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9%.\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_number(reply: str) -> float | None:
    r = reply.replace("≈", "").replace("~", "")
    # prefer a number followed by a percent sign or "profiles", else the first number
    m = re.search(r"(-?\d[\d,]*\.?\d*)\s*(%|percent|profiles|people)", r, re.I)
    txt = m.group(1) if m else None
    if txt is None:
        m2 = _NUM.search(r)
        txt = m2.group(0) if m2 else None
    if txt is None:
        return None
    try:
        return float(txt.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_label(reply: str, options: list[str]) -> str | None:
    """The option that occurs first in the reply (longest option wins at the same position)."""
    r = norm(reply)
    best = None
    for opt in sorted(options, key=len, reverse=True):
        o = norm(opt)
        k = r.find(o)
        if k >= 0 and (best is None or k < best[0]):
            best = (k, opt)
    return best[1] if best else None


def extract_yesno(reply: str) -> str | None:
    r = norm(reply)
    m = re.search(r"\b(yes|no)\b", r)
    return m.group(1) if m else None


def extract_trend(reply: str) -> str | None:
    r = norm(reply)
    pos = []
    for key, val in (("increas", "increase"), ("rise", "increase"), ("grow", "increase"), ("higher", "increase"),
                     ("decreas", "decrease"), ("fall", "decrease"), ("drop", "decrease"), ("lower", "decrease"),
                     ("declin", "decrease"), ("stay the same", "stay the same"), ("same", "stay the same"),
                     ("unchanged", "stay the same"), ("constant", "stay the same")):
        k = r.find(key)
        if k >= 0:
            pos.append((k, val))
    return min(pos)[1] if pos else None


def score(q: dict[str, Any], reply: str) -> dict[str, Any]:
    """Score one reply. Returns the extracted prediction, whether it is correct (relaxed) and exactly correct."""
    at, gold = q["answer_type"], q["answer"]
    if at == "label":
        pred = extract_label(reply, q["options"])
        ok = pred is not None and norm(pred) == norm(gold)
        return {"pred": pred, "correct": ok, "exact": ok, "parsed": pred is not None}
    if at == "yesno":
        pred = extract_yesno(reply)
        ok = pred == gold
        return {"pred": pred, "correct": ok, "exact": ok, "parsed": pred is not None}
    if at == "trend":
        pred = extract_trend(reply)
        ok = pred == gold
        return {"pred": pred, "correct": ok, "exact": ok, "parsed": pred is not None}
    pred = extract_number(reply)
    if pred is None:
        return {"pred": None, "correct": False, "exact": False, "parsed": False}
    g = float(gold)
    tol = max(0.05 * abs(g), 1.0)
    return {"pred": pred, "correct": abs(pred - g) <= tol, "exact": abs(pred - g) < 1e-9, "parsed": True}


def chance_level(q: dict[str, Any]) -> float:
    """Expected accuracy of a uniformly random choice among the options (0 for open numbers)."""
    at = q["answer_type"]
    if at in ("label", "yesno", "trend"):
        return 1.0 / max(1, len(q["options"]))
    return 0.0
