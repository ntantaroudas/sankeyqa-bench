"""Question templates whose answers are computed exactly from the flow table.

Every question carries its answer, its answer type (label, number, integer, yes/no or trend) and ``needs``, the
information a reader must be able to see in the picture to answer it exactly rather than by estimation
("labels" the node names, "node_values" the number printed after each node name, "link_values" the numbers printed
on the ribbons, "title" the title with the total number of profiles).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .cohort import Cohort, STAGES, OTHER

QTYPES = ["top_node", "node_count", "node_value", "share_pct", "max_outflow", "link_compare", "greedy_path",
          "flow_exists", "link_value", "trend", "total"]


@dataclasses.dataclass
class Question:
    qtype: str
    text: str
    answer: Any
    answer_type: str            # "label", "integer", "number", "yesno", "trend"
    options: list[str]          # candidate labels for label questions (empty otherwise)
    needs: list[str]
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _entity(c: Cohort) -> tuple[str, str]:
    return ("role", "roles") if c.config.map_type == "role" else ("industry", "industries")


def _stage_name(s: int) -> str:
    return STAGES[s]


def _q_top_node(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(2, 4))
    labs = c.node_labels(s)
    if len(labs) < 2:
        return None
    vals = [c.nodes[s][l] for l in labs]
    if vals[0] == vals[1]:
        return None
    e, _ = _entity(c)
    return Question("top_node", f"Which {e} has the most profiles at the {_stage_name(s)} stage?", labs[0], "label",
                    labs, ["labels"], {"stage": s, "margin": (vals[0] - vals[1]) / c.n_profiles})


def _q_node_count(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(1, 4))
    _, es = _entity(c)
    return Question("node_count", f"How many different {es} are shown at the {_stage_name(s)} stage, counting "
                    f"\"Other\" as one if it appears?", len(c.nodes[s]), "integer", [], ["labels"], {"stage": s})


def _q_node_value(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(1, 4))
    labs = c.node_labels(s)
    lab = str(rng.choice(labs))
    return Question("node_value", f"How many profiles are {lab} at the {_stage_name(s)} stage?", c.nodes[s][lab],
                    "number", [], ["labels", "node_values"], {"stage": s, "label": lab,
                                                              "share": c.nodes[s][lab] / c.n_profiles})


def _q_share_pct(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(1, 4))
    labs = c.node_labels(s)
    lab = str(rng.choice(labs))
    pct = round(100.0 * c.nodes[s][lab] / c.n_profiles, 1)
    return Question("share_pct", f"What percentage of all profiles are {lab} at the {_stage_name(s)} stage? "
                    f"Answer with a number.", pct, "number", [], ["labels"],
                    {"stage": s, "label": lab, "share": c.nodes[s][lab] / c.n_profiles})


def _movers(c: Cohort, s: int, lab: str) -> list[tuple[str, int]]:
    """Outgoing links of a node to a different label, largest first."""
    return [(t, n) for t, n in c.links_from(s, lab) if t != lab]


def _q_max_outflow(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(0, 3))
    cands = [l for l in c.node_labels(s) if len(_movers(c, s, l)) >= 2]
    if not cands:
        return None
    lab = str(rng.choice(cands))
    outs = _movers(c, s, lab)
    if outs[0][1] == outs[1][1]:
        return None
    e, _ = _entity(c)
    return Question("max_outflow", f"Of the profiles that are {lab} at the {_stage_name(s)} stage and are in a "
                    f"different {e} at the {_stage_name(s + 1)} stage, which {e} do most of them move to?",
                    outs[0][0], "label", [t for t, _ in outs], ["labels"],
                    {"stage": s, "label": lab, "n_out": len(outs),
                     "margin": (outs[0][1] - outs[1][1]) / c.nodes[s][lab]})


def _q_link_compare(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(0, 3))
    cands = [l for l in c.node_labels(s) if len(c.links_from(s, l)) >= 2]
    if not cands:
        return None
    lab = str(rng.choice(cands))
    outs = c.links_from(s, lab)
    i, j = rng.choice(len(outs), size=2, replace=False)
    (b, cb), (d, cd) = outs[int(i)], outs[int(j)]
    if cb == cd:
        return None
    larger = b if cb > cd else d
    return Question("link_compare", f"Between the {_stage_name(s)} and {_stage_name(s + 1)} stages, which is larger, "
                    f"the flow from {lab} to {b} or the flow from {lab} to {d}? Answer with the name of the "
                    f"destination.", larger, "label", [b, d], ["labels"],
                    {"stage": s, "label": lab, "ratio": max(cb, cd) / min(cb, cd)})


def _q_greedy_path(c: Cohort, rng) -> Question | None:
    s0 = int(rng.integers(0, 2))
    labs = c.node_labels(s0)
    lab = str(rng.choice(labs))
    cur = lab
    hops = 0
    for s in range(s0, 3):
        outs = _movers(c, s, cur)
        if not outs:
            return None
        if len(outs) > 1 and outs[0][1] == outs[1][1]:
            return None
        cur = outs[0][0]
        hops += 1
    e, _ = _entity(c)
    return Question("greedy_path", f"Start from {lab} at the {_stage_name(s0)} stage. At every stage follow the "
                    f"largest flow that goes to a different {e}. Which {e} do you reach at the 10 years stage?",
                    cur, "label", c.node_labels(3), ["labels"], {"start_stage": s0, "label": lab, "hops": hops})


def _q_flow_exists(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(0, 3))
    a = str(rng.choice(c.node_labels(s)))
    want_yes = bool(rng.random() < 0.4)
    outs = dict(c.links_from(s, a))
    tgts = c.node_labels(s + 1)
    pool = [t for t in tgts if (t in outs) == want_yes]
    if not pool:
        return None
    b = str(rng.choice(pool))
    cnt = outs.get(b, 0)
    return Question("flow_exists", f"Is there any flow from {a} at the {_stage_name(s)} stage to {b} at the "
                    f"{_stage_name(s + 1)} stage? Answer yes or no.", "yes" if want_yes else "no", "yesno",
                    ["yes", "no"], ["labels"], {"stage": s, "source": a, "target": b, "count": cnt,
                                                "share": cnt / c.n_profiles})


def _q_link_value(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(0, 3))
    links = [(a, b, cnt) for s_, a, b, cnt in c.links if s_ == s]
    a, b, cnt = links[int(rng.integers(0, len(links)))]
    return Question("link_value", f"How many profiles move from {a} at the {_stage_name(s)} stage to {b} at the "
                    f"{_stage_name(s + 1)} stage?", cnt, "number", [], ["labels", "link_values"],
                    {"stage": s, "source": a, "target": b, "share": cnt / c.n_profiles})


def _q_trend(c: Cohort, rng) -> Question | None:
    s = int(rng.integers(1, 3))
    labs = sorted(set(c.node_labels(s)) | set(c.node_labels(s + 1)))
    start = c.config.start_role if c.config.map_type == "role" else c.config.industry
    # the starting label shrinks while the others grow, so it is drawn more often to balance the answers
    w = np.array([3.0 if l == start else 1.0 for l in labs])
    lab = str(rng.choice(labs, p=w / w.sum()))
    v0, v1 = c.nodes[s].get(lab, 0), c.nodes[s + 1].get(lab, 0)
    ans = "increase" if v1 > v0 else ("decrease" if v1 < v0 else "stay the same")
    return Question("trend", f"Does the number of {lab} profiles increase, decrease or stay the same between the "
                    f"{_stage_name(s)} and {_stage_name(s + 1)} stages?", ans, "trend",
                    ["increase", "decrease", "stay the same"], ["labels"],
                    {"stage": s, "label": lab, "from": v0, "to": v1, "rel_change": (v1 - v0) / max(1, v0)})


def _q_total(c: Cohort, rng) -> Question | None:
    return Question("total", "How many profiles does this career map include in total?", c.n_profiles, "integer",
                    [], ["title"], {})


GENERATORS = {
    "top_node": _q_top_node, "node_count": _q_node_count, "node_value": _q_node_value, "share_pct": _q_share_pct,
    "max_outflow": _q_max_outflow, "link_compare": _q_link_compare, "greedy_path": _q_greedy_path,
    "flow_exists": _q_flow_exists, "link_value": _q_link_value, "trend": _q_trend, "total": _q_total,
}


def make_questions(c: Cohort, seed: int, per_type: int = 1, qtypes: list[str] | None = None,
                   tries: int = 12) -> list[Question]:
    """``per_type`` questions of each type for a cohort, skipping templates that do not apply (ties, a single
    node). Questions of the same type and the same text are not repeated."""
    rng = np.random.default_rng(seed)
    out: list[Question] = []
    for qt in qtypes or QTYPES:
        seen = set()
        got = 0
        for _ in range(tries * per_type):
            q = GENERATORS[qt](c, rng)
            if q is None or q.text in seen:
                continue
            seen.add(q.text)
            out.append(q)
            got += 1
            if got >= per_type:
                break
    return out


def answer_from_table(c: Cohort, q: Question) -> Any:
    """Recompute the answer of a question from the flow table (the exact engine). Used as the upper bound and to
    check that the stored answers are consistent with the table."""
    m, qt = q.meta, q.qtype
    if qt == "top_node":
        return c.node_labels(m["stage"])[0]
    if qt == "node_count":
        return len(c.nodes[m["stage"]])
    if qt == "node_value":
        return c.nodes[m["stage"]][m["label"]]
    if qt == "share_pct":
        return round(100.0 * c.nodes[m["stage"]][m["label"]] / c.n_profiles, 1)
    if qt == "max_outflow":
        return _movers(c, m["stage"], m["label"])[0][0]
    if qt == "link_compare":
        a, b, d = m["label"], q.options[0], q.options[1]
        return b if c.link_count(m["stage"], a, b) > c.link_count(m["stage"], a, d) else d
    if qt == "greedy_path":
        cur = m["label"]
        for s in range(m["start_stage"], 3):
            cur = _movers(c, s, cur)[0][0]
        return cur
    if qt == "flow_exists":
        return "yes" if c.link_count(m["stage"], m["source"], m["target"]) > 0 else "no"
    if qt == "link_value":
        return c.link_count(m["stage"], m["source"], m["target"])
    if qt == "trend":
        v0, v1 = c.nodes[m["stage"]].get(m["label"], 0), c.nodes[m["stage"] + 1].get(m["label"], 0)
        return "increase" if v1 > v0 else ("decrease" if v1 < v0 else "stay the same")
    if qt == "total":
        return c.n_profiles
    raise ValueError(qt)
