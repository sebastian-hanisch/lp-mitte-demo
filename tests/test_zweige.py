"""Zweig-Tests (Nullspalten-Regel, DEMO-PLAYBOOK §0): jede Variante weicht auf mindestens einer Instanz von der Nachbarvariante ab."""

import math
from dataclasses import replace

import numpy as np
import pytest

import lpm_center as K
import lpm_constants as C
import lpm_evaluation as V
import lpm_regret as RG
from lpm_scenario import Draws, layout_batch, make_points


def test_undamped_and_damped_iteration_differ_at_p_four():
    base = V.Settings(p=4.0, solver=C.SOLVERS[1])
    plain, damped = V.analyse(base), V.analyse(replace(base, solver=C.SOLVERS[2]))
    assert plain.solve.status == "maxit" and damped.solve.status == "ok" and plain.solver_gap > 1e-3 > damped.solver_gap


def test_all_three_solvers_differ_in_their_iteration_counts_but_reach_the_same_center():
    base = V.Settings(p=3.0)
    runs = {s: V.analyse(replace(base, solver=s)) for s in C.SOLVERS}
    counts = {s: r.solve.iterations for s, r in runs.items()}
    assert len(set(counts.values())) == 3
    for r in runs.values():
        assert r.solve.status == "ok" and r.solver_gap < 1e-5


def test_the_iteration_against_newton_differs_on_the_limit_instance_only_without_damping():
    pts = make_points("Normal", 30, 7)
    start = K.centroid(pts)
    newton = K.newton_lp(pts, 8.0, start).center
    assert np.linalg.norm(K.irls(pts, 8.0, start, maxit=2000).center - newton) > 0.1
    assert np.linalg.norm(K.irls(pts, 8.0, start, relax=K.damping(8.0), maxit=2000).center - newton) < 1e-5


def test_random_start_changes_the_first_iterate_and_the_end_for_p_below_one():
    base = V.Settings(layout="Zwei Häufungen", n=100, share=60, sep=8.0, spread=3.0, p=0.4, seed=8)
    pts = V.build_points(base)
    assert not np.allclose(V.start_point(base, pts), V.start_point(replace(base, start=C.STARTS[1]), pts))
    ends = {tuple(np.round(V.analyse(replace(base, start=s, seed=sd)).center, 3)) for s in C.STARTS for sd in (8, 9, 10, 11)}
    assert len(ends) >= 4                                          # verschiedene Starts enden in verschiedenen lokalen Minima
    for p in (3.0, 1.0):
        centers = [V.analyse(V.Settings(p=p, start=s)).center for s in C.STARTS]
        assert np.allclose(centers[0], centers[1], atol=1e-5)      # für p ≥ 1 führt jeder Start zur selben Mitte


def test_outpost_distance_moves_the_one_center_and_the_regrets():
    near, far = make_points("Außenposten", 30, 7, outpost=1.0), make_points("Außenposten", 30, 7, outpost=12.0)
    assert np.linalg.norm(K.one_center(near)[0] - K.one_center(far)[0]) > 3.0
    assert RG.regret_matrix(far)[2, 1] > RG.regret_matrix(near)[2, 1] + 50


def test_share_distance_and_spread_each_change_the_regret_matrix():
    base = make_points("Zwei Häufungen", 60, 7)
    for kw in ({"share": 0.9}, {"sep": 10.0}, {"spread": 3.0}):
        other = make_points("Zwei Häufungen", 60, 7, **kw)
        assert abs(RG.regret_matrix(other) - RG.regret_matrix(base)).max() > 1.0, kw


def test_the_criterion_changes_the_verdict_and_the_highlighted_metric():
    texts = {c: V.verdict_regret(V.analyse(V.Settings(layout="Außenposten", criterion=c)))[1] for c in C.CRITERIA}
    assert len(set(texts.values())) == 3


def test_every_exponent_regime_runs_its_own_solver():
    pts = make_points("Normal", 30, 7)
    status = {p: V.run_solver(V.Settings(p=p), pts) for p in (0.5, 1.0, 2.0, 3.0, math.inf)}
    assert status[2.0].iterations == 0 and status[math.inf].iterations == 1 and status[3.0].iterations > 0 and status[1.0].iterations > 5 and status[0.5].iterations > 0
    assert len({tuple(np.round(r.center, 4)) for p, r in status.items() if p != 0.5}) == 4
    assert np.linalg.norm(pts - status[0.5].center, axis=1).min() < 1e-6          # p < 1: der Lauf endet auf einem Kunden


def test_p_below_one_finds_the_other_cluster_where_p_one_does_not():
    """H8 an einer Instanz mit zwei Dichten: p = 1 bleibt in der Nähe der breiten Häufung, p = 0,4 wählt einen Kunden der dichten."""
    s = V.Settings(layout="Zwei Häufungen", n=100, share=60, sep=8.0, spread=3.0, seed=8)
    one, below = V.analyse(replace(s, p=1.0)), V.analyse(replace(s, p=0.4))
    assert one.reference[0] < 4.0 < below.reference[0] and below.reference_on_data and one.reference_on_data is None


def test_ties_of_the_axis_instance_follow_the_documented_rule(axis_ties):
    assert K.solve_1d(axis_ties, 0.5)[0] == 1.0 and K.solve_1d(axis_ties[::-1], 0.5)[0] == 1.0


def test_each_layout_shifts_the_centroid_and_the_one_center_differently():
    draws = {layout: layout_batch(Draws(5), layout, 1, 30)[0] for layout in C.LAYOUTS}
    radii = {layout: round(K.one_center(pts)[1], 3) for layout, pts in draws.items()}
    assert len(set(radii.values())) == 4
