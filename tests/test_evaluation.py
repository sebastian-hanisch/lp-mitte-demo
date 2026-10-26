"""Analyse einer Einstellung: die angezeigten Kennzahlen gegen unabhängige Rechnung (Mitten, Kosten, Regrets, Pfad)."""

import math

import numpy as np
import pytest

import lpm_center as K
import lpm_constants as C
import lpm_evaluation as V
from lpm_scenario import make_points


def _cost(points, c, q):
    d = np.linalg.norm(points - c, axis=1)
    return d.max() if q == math.inf else (d ** q).sum()


def test_standard_analysis_marks_and_costs():
    s = V.Settings()
    a = V.analyse(s)
    pts = make_points("Normal", 30, 7)
    assert a.points == pytest.approx(pts) and a.centroid == pytest.approx(pts.mean(axis=0))
    # das Minimum der Summe der Wege liegt im geometrischen Median: jede kleine Verschiebung erhöht sie
    for d in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert _cost(pts, a.median + d, 1) > _cost(pts, a.median, 1)
    assert a.one_radius == pytest.approx(np.linalg.norm(pts - a.one_center, axis=1).max())
    assert a.costs == pytest.approx([_cost(pts, a.reference, q) for q in (1, 2, math.inf)])
    assert a.optima == pytest.approx([_cost(pts, a.median, 1), _cost(pts, a.centroid, 2), a.one_radius])
    assert a.regrets == pytest.approx(100 * (a.costs / a.optima - 1)) and a.regrets[1] == pytest.approx(0.0, abs=1e-9)
    assert a.nearest == pytest.approx(np.linalg.norm(pts - a.reference, axis=1).min())
    assert a.cost_p == pytest.approx(a.costs[1])


def test_path_runs_from_the_median_through_the_centroid_to_the_one_center():
    a = V.analyse(V.Settings(layout="Außenposten", p=3.0))
    assert a.path_p == C.PATH_P + (math.inf,) and a.path.shape == (len(C.PATH_P) + 1, 2)
    assert a.path[0] == pytest.approx(a.median) and a.path[C.PATH_P.index(2.0)] == pytest.approx(a.centroid) and a.path[-1] == pytest.approx(a.one_center)
    assert a.path[C.PATH_P.index(3.0)] == pytest.approx(a.reference)
    assert a.path[C.PATH_P.index(4.0)] == pytest.approx(K.newton_lp(a.points, 4.0).center)


def test_reference_and_solver_agree_for_p_above_one_and_report_on_data_below_one():
    for p in (1.0, 1.5, 2.0, 5.0, 64.0, math.inf):
        a = V.analyse(V.Settings(p=p, solver=C.SOLVERS[2]))
        assert a.solver_gap < 1e-4 and a.reference_on_data is None, p
    low = V.analyse(V.Settings(p=0.3))
    assert low.reference_on_data is True and np.linalg.norm(low.points - low.reference, axis=1).min() < 1e-9 and low.nearest < 1e-9


def test_start_point_is_deterministic_and_inside_the_bounding_box():
    s = V.Settings(start=C.STARTS[1], seed=11)
    pts = V.build_points(s)
    start = V.start_point(s, pts)
    assert (start >= pts.min(axis=0)).all() and (start <= pts.max(axis=0)).all() and start == pytest.approx(V.start_point(s, pts))
    assert V.start_point(V.Settings(start=C.STARTS[0]), pts) == pytest.approx(pts.mean(axis=0))
    assert not np.allclose(start, V.start_point(V.Settings(start=C.STARTS[1], seed=12), pts))


def test_criterion_q_maps_every_label():
    assert [V.criterion_q(c) for c in C.CRITERIA] == [1.0, 2.0, math.inf]


def test_run_solver_reports_each_regime():
    pts = make_points("Normal", 30, 7)
    one = V.run_solver(V.Settings(p=1.0), pts)
    assert one.status in ("ok", "datenpunkt") and one.iterations > 5 and one.center == pytest.approx(K.vardi_zhang(pts, np.ones(30), tol=1e-12).center, abs=1e-6)
    assert V.run_solver(V.Settings(p=math.inf), pts).center == pytest.approx(K.one_center(pts)[0])
    assert V.run_solver(V.Settings(p=2.0, solver=C.SOLVERS[1]), pts).iterations == 0                     # für p = 2 gilt die Formel, gleich welcher Löser
    for solver in C.SOLVERS:
        r = V.run_solver(V.Settings(p=2.5, solver=solver), pts)
        assert r.status == "ok" and r.center == pytest.approx(K.newton_lp(pts, 2.5).center, abs=1e-6)
    low = V.run_solver(V.Settings(p=0.5, solver=C.SOLVERS[0]), pts)                                     # p < 1: immer die Gewichtsiteration
    assert low.status == "ok" and low.trace is None
