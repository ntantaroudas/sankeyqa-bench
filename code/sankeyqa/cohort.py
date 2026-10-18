"""Synthetic career cohorts and the flow tables behind career-map Sankey diagrams.

A career map in XR-CareerAssist starts from a questionnaire (current role, years of experience, industry, age group
and a target role), matches anonymised professional profiles with similar characteristics, and shows how those
profiles moved between roles, or between industries, after 2, 5 and 10 years. The real platform draws on a private
database of profiles. This module instead simulates a cohort for every questionnaire configuration with a small
Markov model whose parameters depend on the configuration, so that the diagrams have the same structure as the real
ones while every number behind them is known exactly and nothing private is used.

Nothing here is a claim about real careers. The cohorts exist so that the questions have exact answers.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np

STAGES = ["Now", "2 years", "5 years", "10 years"]
STAGE_YEARS = [0, 2, 5, 10]

# Six roles on a ladder (rank 0 to 5) and two lateral roles that people move into from any rank.
LADDER = ["Employee", "Specialist", "Team Lead", "Manager", "Director", "Executive"]
LATERAL = ["Consultant", "Founder"]
ROLES = LADDER + LATERAL

INDUSTRIES = ["IT Services", "Software Products", "Computer Manufacturing", "Financial Services",
              "Telecommunications", "Consulting", "Healthcare", "Education", "Public Sector", "Retail",
              "Energy", "Manufacturing"]
# Industries that people move between more often (symmetric neighbourhoods).
RELATED = {
    "IT Services": ["Software Products", "Telecommunications", "Consulting"],
    "Software Products": ["IT Services", "Computer Manufacturing", "Consulting"],
    "Computer Manufacturing": ["Software Products", "Manufacturing", "Telecommunications"],
    "Financial Services": ["Consulting", "IT Services", "Retail"],
    "Telecommunications": ["IT Services", "Computer Manufacturing", "Energy"],
    "Consulting": ["Financial Services", "IT Services", "Public Sector"],
    "Healthcare": ["Public Sector", "Education", "Consulting"],
    "Education": ["Public Sector", "Healthcare", "Consulting"],
    "Public Sector": ["Education", "Healthcare", "Consulting"],
    "Retail": ["Financial Services", "Manufacturing", "Consulting"],
    "Energy": ["Manufacturing", "Telecommunications", "Public Sector"],
    "Manufacturing": ["Energy", "Computer Manufacturing", "Retail"],
}
EXPERIENCE = [2, 5, 10, 15, 20, 25]
AGE_GROUPS = ["21-30", "31-40", "41-50", "51-60"]
OTHER = "Other"


@dataclasses.dataclass
class Config:
    """One questionnaire configuration, the input of a career map."""
    map_type: str            # "role" or "industry"
    start_role: str
    years_experience: int
    industry: str
    age_group: str
    target_role: str
    n_profiles: int
    max_nodes: int           # visible nodes per stage, the rest are merged into "Other"

    def key(self) -> str:
        s = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Cohort:
    """A cohort and its flow table. ``nodes[s]`` maps a label to a count at stage ``s``; ``links`` are
    (stage index of the source, source label, target label, count) for the transition from stage s to s+1."""
    config: Config
    nodes: list[dict[str, int]]
    links: list[tuple[int, str, str, int]]
    n_profiles: int

    def table(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(), "stages": STAGES, "n_profiles": self.n_profiles,
            "nodes": [[{"label": k, "count": int(v)} for k, v in sorted(st.items(), key=lambda kv: (-kv[1], kv[0]))]
                      for st in self.nodes],
            "links": [{"stage": s, "source": a, "target": b, "count": int(c)} for s, a, b, c in self.links],
        }

    @staticmethod
    def from_table(t: dict[str, Any]) -> "Cohort":
        cfg = Config(**t["config"])
        nodes = [{d["label"]: int(d["count"]) for d in st} for st in t["nodes"]]
        links = [(int(l["stage"]), l["source"], l["target"], int(l["count"])) for l in t["links"]]
        return Cohort(cfg, nodes, links, int(t["n_profiles"]))

    def node_labels(self, stage: int) -> list[str]:
        """Labels at a stage, largest first."""
        return [k for k, _ in sorted(self.nodes[stage].items(), key=lambda kv: (-kv[1], kv[0]))]

    def links_from(self, stage: int, label: str) -> list[tuple[str, int]]:
        return sorted([(b, c) for s, a, b, c in self.links if s == stage and a == label], key=lambda x: (-x[1], x[0]))

    def links_into(self, stage: int, label: str) -> list[tuple[str, int]]:
        """Links arriving at ``label`` at stage ``stage`` (from stage-1)."""
        return sorted([(a, c) for s, a, b, c in self.links if s == stage - 1 and b == label], key=lambda x: (-x[1], x[0]))

    def link_count(self, stage: int, a: str, b: str) -> int:
        for s, x, y, c in self.links:
            if s == stage and x == a and y == b:
                return c
        return 0


def _rng(seed: int | str) -> np.random.Generator:
    if isinstance(seed, str):
        seed = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def sample_config(rng: np.random.Generator, map_type: str | None = None) -> Config:
    """A random questionnaire configuration."""
    mt = map_type or ("role" if rng.random() < 0.6 else "industry")
    start = str(rng.choice(LADDER[:5], p=[0.30, 0.20, 0.15, 0.25, 0.10]))
    years = int(rng.choice(EXPERIENCE))
    # age loosely tied to experience
    age = AGE_GROUPS[min(3, max(0, int((years + rng.integers(-3, 4)) // 8)))]
    rank = LADDER.index(start)
    target = str(rng.choice(LADDER[min(5, rank + 1):] or [LADDER[5]]))
    n = int(rng.choice([60, 90, 120, 180, 240, 320, 450, 600, 800, 1200]))
    return Config(mt, start, years, str(rng.choice(INDUSTRIES)), age, target, n, int(rng.choice([4, 5, 6, 7])))


def _role_step_matrix(cfg: Config, rng: np.random.Generator, years: float) -> np.ndarray:
    """Transition probabilities between the 8 roles over ``years`` years for this cohort."""
    k = len(ROLES)
    exp_factor = 1.0 / (1.0 + cfg.years_experience / 12.0)          # less experience, more movement
    age_factor = {"21-30": 1.15, "31-40": 1.0, "41-50": 0.8, "51-60": 0.6}[cfg.age_group]
    # probability of leaving the current role over the step falls off exponentially with time
    lam = 0.22 * (0.6 + 0.8 * exp_factor) * age_factor * rng.uniform(0.75, 1.3)
    mobility = min(0.92, 1.0 - float(np.exp(-lam * years)))
    target_rank = LADDER.index(cfg.target_role)
    M = np.zeros((k, k))
    for i, r in enumerate(ROLES):
        if r in LATERAL:
            # lateral roles are fairly sticky, some return to the ladder
            back = 0.25 * rng.uniform(0.6, 1.4)
            M[i, i] = 1 - back
            for j in range(len(LADDER)):
                M[i, j] = back * np.exp(-0.6 * abs(j - 3))
            M[i, :len(LADDER)] *= back / M[i, :len(LADDER)].sum()
            continue
        stay = 1 - mobility
        up = mobility * 0.72
        down = mobility * 0.10
        lat = mobility * 0.18
        M[i, i] = stay
        # upward moves, pulled towards the target rank
        ups = [j for j in range(i + 1, len(LADDER))]
        if ups:
            w = np.array([np.exp(-0.9 * (j - i - 1)) * (1.6 if j <= target_rank else 0.5) for j in ups])
            M[i, ups] += up * w / w.sum()
        else:
            M[i, i] += up
        if i > 0:
            M[i, i - 1] += down
        else:
            M[i, i] += down
        M[i, len(LADDER):] += lat * np.array([0.65, 0.35])
    # per-cohort noise
    M = M * rng.uniform(0.85, 1.15, size=M.shape)
    M /= M.sum(axis=1, keepdims=True)
    return M


def _industry_step_matrix(cfg: Config, rng: np.random.Generator, years: float, pool: list[str]) -> np.ndarray:
    k = len(pool)
    mu = 0.08 * rng.uniform(0.6, 1.5)
    stick = max(0.5, float(np.exp(-mu * years)))
    M = np.zeros((k, k))
    for i, a in enumerate(pool):
        M[i, i] = stick
        w = np.array([0.0 if j == i else (3.0 if pool[j] in RELATED[a] else 1.0) for j in range(k)])
        w *= rng.uniform(0.7, 1.3, size=k)
        if w.sum() > 0:
            M[i] += (1 - stick) * w / w.sum()
        else:
            M[i, i] = 1.0
    return M


def simulate(cfg: Config, seed: int | str | None = None) -> Cohort:
    """Simulate the cohort of a configuration and aggregate it into a flow table with at most ``max_nodes``
    visible labels per stage (the rest merged into "Other")."""
    rng = _rng(seed if seed is not None else cfg.key())
    n = cfg.n_profiles
    if cfg.map_type == "role":
        labels = ROLES
        start_idx = np.full(n, ROLES.index(cfg.start_role))
        mats = [_role_step_matrix(cfg, rng, STAGE_YEARS[s + 1] - STAGE_YEARS[s]) for s in range(3)]
    else:
        pool_size = int(rng.integers(4, 8))
        others = [x for x in INDUSTRIES if x != cfg.industry]
        rel = [x for x in RELATED[cfg.industry] if x in others]
        rest = [x for x in others if x not in rel]
        pool = [cfg.industry] + rel[:pool_size - 1]
        while len(pool) < pool_size:
            pool.append(rest.pop(int(rng.integers(0, len(rest)))))
        labels = pool
        start_idx = np.full(n, 0)
        mats = [_industry_step_matrix(cfg, rng, STAGE_YEARS[s + 1] - STAGE_YEARS[s], pool) for s in range(3)]
    paths = np.zeros((n, 4), dtype=int)
    paths[:, 0] = start_idx
    for s in range(3):
        M = mats[s]
        cum = np.cumsum(M, axis=1)
        u = rng.random(n)
        cur = paths[:, s]
        nxt = (u[:, None] > cum[cur]).sum(axis=1)
        paths[:, s + 1] = np.minimum(nxt, len(labels) - 1)
    # visible labels per stage, the rest merged into Other
    nodes: list[dict[str, int]] = []
    vis: list[dict[int, str]] = []
    for s in range(4):
        counts = np.bincount(paths[:, s], minlength=len(labels))
        order = np.argsort(-counts, kind="stable")
        keep = [int(i) for i in order[:cfg.max_nodes] if counts[i] > 0]
        mapping = {int(i): (labels[int(i)] if int(i) in keep else OTHER) for i in range(len(labels))}
        st: dict[str, int] = {}
        for i, c in enumerate(counts):
            if c > 0:
                st[mapping[i]] = st.get(mapping[i], 0) + int(c)
        nodes.append(st)
        vis.append(mapping)
    links: dict[tuple[int, str, str], int] = {}
    for s in range(3):
        for p in paths:
            key = (s, vis[s][int(p[s])], vis[s + 1][int(p[s + 1])])
            links[key] = links.get(key, 0) + 1
    link_list = sorted(((s, a, b, c) for (s, a, b), c in links.items()), key=lambda x: (x[0], -x[3], x[1], x[2]))
    return Cohort(cfg, nodes, link_list, n)


def check_consistency(c: Cohort) -> None:
    """Every node total equals the sum of its outgoing links (stages 0 to 2) and of its incoming links (1 to 3)."""
    for s in range(3):
        for lab, cnt in c.nodes[s].items():
            assert sum(x for _, x in c.links_from(s, lab)) == cnt, (s, lab)
    for s in range(1, 4):
        for lab, cnt in c.nodes[s].items():
            assert sum(x for _, x in c.links_into(s, lab)) == cnt, (s, lab)
    assert all(sum(st.values()) == c.n_profiles for st in c.nodes)
