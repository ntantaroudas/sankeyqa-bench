"""Tests of the benchmark generator and the scorer (no model downloads, run with ``python -m pytest tests``)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sankeyqa.cohort import sample_config, simulate, check_consistency, Cohort  # noqa: E402
from sankeyqa.questions import make_questions, answer_from_table, QTYPES  # noqa: E402
from sankeyqa.render import render, RenderStyle  # noqa: E402
from sankeyqa.scoring import score  # noqa: E402


def test_flow_tables_are_consistent_and_deterministic():
    rng = np.random.default_rng(3)
    for _ in range(40):
        cfg = sample_config(rng)
        a, b = simulate(cfg), simulate(cfg)
        check_consistency(a)
        assert a.table() == b.table()
        assert Cohort.from_table(a.table()).table() == a.table()


def test_every_answer_matches_the_table_and_scores_correct():
    rng = np.random.default_rng(4)
    seen = set()
    for i in range(40):
        c = simulate(sample_config(rng))
        for q in make_questions(c, seed=i):
            seen.add(q.qtype)
            assert answer_from_table(c, q) == q.answer
            assert score(q.as_dict(), f"The answer is {q.answer}.")["correct"]
    assert seen == set(QTYPES)


def test_scorer_rejects_wrong_answers():
    q = {"answer_type": "number", "answer": 100, "options": []}
    assert score(q, "104")["correct"] and not score(q, "106")["correct"]
    q = {"answer_type": "label", "answer": "Manager", "options": ["Manager", "Team Lead"]}
    assert not score(q, "Team Lead")["correct"]
    assert not score(q, "I do not know")["parsed"]


def test_render_styles():
    c = simulate(sample_config(np.random.default_rng(5)))
    assert render(c, RenderStyle(long_side_px=512)).size[0] == 512
    assert render(c, RenderStyle(background="transparent")).mode == "RGBA"
