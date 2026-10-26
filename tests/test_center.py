"""Die Einheiten von lpm_center.py an den Handrechnungen aus conftest.py (DEMO-PLAYBOOK §1b)."""

import math

import numpy as np
import pytest

import lpm_center as K
from conftest import SQRT47, axis_a_center


def test_lp_cost_for_each_exponent(right_triangle):
    c = np.array([2.0, 1.5])
    assert K.lp_cost(right_triangle, c, 1) == pytest.approx(7.5)
    assert K.lp_cost(right_triangle, c, 2) == pytest.approx(18.75)
    assert K.lp_cost(right_triangle, c, math.inf) == pytest.approx(2.5)
    g = K.centroid(right_triangle)
    assert K.lp_cost(right_triangle, g, 1) == pytest.approx(5 / 3 + math.sqrt(73) / 3 + math.sqrt(52) / 3)
    assert K.lp_cost(right_triangle, g, 2) == pytest.approx(150 / 9)
    assert K.lp_cost(right_triangle, g, math.inf) == pytest.approx(math.sqrt(73) / 3)
    assert K.lp_cost(right_triangle, g, 0.5) == pytest.approx(math.sqrt(5 / 3) + (73 / 9) ** 0.25 + (52 / 9) ** 0.25)


@pytest.mark.parametrize("p", [1.2, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0])
def test_solve_1d_matches_the_closed_formula(axis_a, p):
    lo, hi = K.solve_1d(axis_a[:, 0], p)
    assert lo == hi == pytest.approx(axis_a_center(p), abs=1e-9)


def test_solve_1d_hand_values(axis_b, axis_a):
    assert K.solve_1d(axis_b, 2.0)[0] == pytest.approx(4.0)
    assert K.solve_1d(axis_b, 3.0)[0] == pytest.approx(-2 + SQRT47)
    assert K.solve_1d(axis_b, math.inf) == (5.5, 5.5)
    assert K.solve_1d(axis_b, 1.0) == (2.0, 3.0)                       # gerade Zahl: Intervall der Mediane
    assert K.solve_1d(axis_a[:, 0], 1.0) == (0.0, 0.0)                  # drei Nullen: der Median ist ein Punkt
    assert K.solve_1d(np.array([5.0, 1.0, 3.0]), 1.0) == (3.0, 3.0)     # ungerade Zahl, unsortiert
    assert axis_a_center(1.5) == pytest.approx(1.2) and axis_a_center(3.0) == pytest.approx(12 / (1 + math.sqrt(3)))


def test_solve_1d_below_one_is_a_data_point_with_the_documented_tie_rule(axis_ties):
    lo, hi = K.solve_1d(axis_ties, 0.5)
    assert lo == hi == 1.0                                              # Gleichstand zwischen 1 und 10: der kleinere Wert
    assert K.lp_cost(axis_ties[:, None], np.array([1.0]), 0.5) == pytest.approx(1 + 3 + math.sqrt(10))
    assert K.lp_cost(axis_ties[:, None], np.array([0.0]), 0.5) == pytest.approx(1 + math.sqrt(10) + math.sqrt(11))
    assert K.solve_1d(axis_ties[::-1], 0.5) == (1.0, 1.0)               # unabhängig von der Reihenfolge der Eingabe


def test_solve_1d_degenerate_and_invalid():
    assert K.solve_1d(np.array([3.0, 3.0, 3.0]), 2.5) == (3.0, 3.0)
    assert K.solve_1d(np.array([7.0]), 1.5) == (7.0, 7.0)
    with pytest.raises(ValueError):
        K.solve_1d(np.array([1.0, 2.0]), 0.0)


@pytest.mark.parametrize("p", [1.2, 1.5, 3.0, 8.0, 30.0])
def test_newton_on_the_axis_matches_the_closed_formula(axis_a, p):
    r = K.newton_lp(axis_a, p)
    assert r.status == "ok" and r.center[0] == pytest.approx(axis_a_center(p), abs=1e-8)


def test_newton_two_points_and_p_two_and_invalid(two_points, right_triangle):
    for p in (1.3, 2.0, 5.0, 20.0):
        assert K.newton_lp(two_points, p).center == pytest.approx([3.0, 0.0], abs=1e-8)
    assert K.newton_lp(right_triangle, 2.0).center == pytest.approx([4 / 3, 1.0], abs=1e-9)
    for bad in (1.0, 0.5, math.inf):
        with pytest.raises(ValueError):
            K.newton_lp(right_triangle, bad)


def test_newton_warm_start_at_the_optimum_reports_ok_without_a_step(right_triangle):
    r = K.newton_lp(right_triangle, 2.0, c0=[4 / 3, 1.0])
    assert r.status == "ok" and r.iterations == 0


def test_newton_all_points_equal_and_iteration_limit():
    same = np.array([[1.0, 2.0]] * 4)
    assert K.newton_lp(same, 3.0).center == pytest.approx([1.0, 2.0])
    assert K.newton_lp(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]]), 3.0, c0=[9.0, 9.0], maxit=1).status == "maxit"


def test_damping_is_one_over_p_minus_one_above_two():
    assert K.damping(2.0) == 1.0 and K.damping(1.5) == 1.0 and K.damping(0.5) == 1.0 and K.damping(1.0) == 1.0
    assert K.damping(3.0) == 0.5 and K.damping(5.0) == 0.25 and K.damping(2.5) == pytest.approx(1 / 1.5)


def test_irls_p_two_is_the_centroid_in_one_step(right_triangle):
    r = K.irls(right_triangle, 2.0, [3.0, 3.0])
    assert r.status == "ok" and r.center == pytest.approx([4 / 3, 1.0]) and r.iterations == 2     # Schritt 1 landet im Schwerpunkt, Schritt 2 bewegt sich nicht mehr


def test_irls_p_one_is_weiszfeld_and_reaches_the_geometric_median():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0], [1.0, 1.0]])
    r = K.irls(pts, 1.0, pts.mean(axis=0), tol=1e-13)
    vz = K.vardi_zhang(pts, np.ones(5), tol=1e-13).center
    assert r.center == pytest.approx(vz, abs=1e-8)


def test_damped_irls_step_on_the_axis_is_exactly_the_newton_step():
    """Eigene Herleitung: auf der Achse ist der Schritt der Gewichtsiteration das (p − 1)-fache des Newton-Schritts -> mit Dämpfung 1/(p − 1) genau der Newton-Schritt."""
    x = np.array([[0.0], [1.0], [4.0], [9.0]])
    for p in (3.0, 4.0, 6.0):
        c = np.array([2.5])
        d = np.abs(c - x[:, 0])
        newton = -(np.sign(c - x[:, 0]) * d ** (p - 1)).sum() * p / (p * (p - 1) * (d ** (p - 2)).sum())
        step = K.irls(x, p, c, relax=K.damping(p), maxit=1).center - c
        assert step[0] == pytest.approx(newton, rel=1e-12)
        undamped = K.irls(x, p, c, relax=1.0, maxit=1).center - c
        assert undamped[0] == pytest.approx((p - 1) * newton, rel=1e-12)


def test_irls_statuses_and_trace(right_triangle):
    ok = K.irls(right_triangle, 3.0, right_triangle.mean(axis=0), relax=0.5, trace=True)
    assert ok.status == "ok" and len(ok.trace) == ok.iterations and ok.trace[0] == pytest.approx(K.lp_cost(right_triangle, right_triangle.mean(axis=0), 3))
    assert K.irls(right_triangle, 6.0, [1.0, 1.0], maxit=7).status == "maxit"
    assert K.irls(right_triangle, 3.0, [1.0, 1.0], trace=False).trace is None


def test_irls_multi_matches_irls_per_start(right_triangle):
    starts = np.array([[0.5, 0.5], [3.0, 2.0], [1.0, 2.5]])
    ends, counts = K.irls_multi(right_triangle, 1.5, starts, tol=1e-10)
    for s, e in zip(starts, ends):
        assert e == pytest.approx(K.irls(right_triangle, 1.5, s, tol=1e-10).center, abs=1e-8)
    assert (counts > 0).all()


def test_one_center_right_triangle_and_degenerate(right_triangle, two_points):
    c, r = K.one_center(right_triangle)
    assert c == pytest.approx([2.0, 1.5]) and r == pytest.approx(2.5)
    c2, r2 = K.one_center(two_points)
    assert c2 == pytest.approx([3.0, 0.0]) and r2 == pytest.approx(math.hypot(4, 4) / 2)
    c1, r1 = K.one_center(np.array([[2.0, 3.0]]))
    assert c1 == pytest.approx([2.0, 3.0]) and r1 == 0.0
    cs, rs = K.one_center(np.array([[1.0, 1.0]] * 3))
    assert rs == 0.0
    cl, rl = K.one_center(np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0], [2.0, 2.0]]))      # kollinear
    assert cl == pytest.approx([1.5, 1.5]) and rl == pytest.approx(3 * math.sqrt(2) / 2)


def test_one_center_does_not_depend_on_the_seed_or_the_order(right_triangle):
    pts = np.vstack([right_triangle, [[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]]])
    c0, r0 = K.one_center(pts, 0)
    for seed in (1, 2, 3):
        c, r = K.one_center(pts, seed)
        assert c == pytest.approx(c0, abs=1e-9) and r == pytest.approx(r0, abs=1e-9)
    c, r = K.one_center(pts[::-1], 5)
    assert c == pytest.approx(c0, abs=1e-9)


def test_vardi_zhang_copy_gives_the_geometric_median_of_the_right_triangle(right_triangle):
    c = K.vardi_zhang(right_triangle, np.ones(3), tol=1e-13).center
    assert 0.1 < c[0] < 1.9 and 0.1 < c[1] < 1.4        # Winkel 90° < 120°: das Optimum liegt im Innern, nicht auf einer Ecke
    from itertools import product
    best = min(K.lp_cost(right_triangle, np.array([x, y]), 1) for x, y in product(np.linspace(0, 2, 201), np.linspace(0, 2, 201)))
    assert K.lp_cost(right_triangle, c, 1) <= best + 1e-9


def test_best_below_one_picks_the_better_cluster_point_and_validates():
    pts = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1], [5.1, 5.1], [5.05, 5.05]])
    r = K.best_below_one(pts, 0.5, seed=3)
    assert r.on_data and np.linalg.norm(r.center - pts[4:], axis=1).min() < 1e-12 and r.center[0] > 4
    assert r.cost == pytest.approx(r.data_cost) and r.ends.shape == (20, 2)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            K.best_below_one(pts, bad)


def test_lp_center_dispatch(right_triangle):
    assert K.lp_center(right_triangle, math.inf) == pytest.approx([2.0, 1.5])
    assert K.lp_center(right_triangle, 2.0) == pytest.approx([4 / 3, 1.0])
    assert K.lp_center(right_triangle, 3.0) == pytest.approx(K.newton_lp(right_triangle, 3.0).center)
    assert K.lp_center(right_triangle, 1.0) == pytest.approx(K.vardi_zhang(right_triangle, np.ones(3), tol=1e-12).center)
    assert K.lp_center(right_triangle, 0.5, seed=1) == pytest.approx(K.best_below_one(right_triangle, 0.5, 1).center)


def test_circle_three_of_collinear_points_is_the_circle_over_the_two_outer_ones():
    c, r = K._circle_three(np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    assert c == pytest.approx([1.5, 1.5]) and r == pytest.approx(3 * math.sqrt(2) / 2)
    c, r = K._circle_three(np.array([0.0, 0.0]), np.array([4.0, 0.0]), np.array([0.0, 3.0]))              # rechtwinkliges Dreieck: Umkreis über der Hypotenuse
    assert c == pytest.approx([2.0, 1.5]) and r == pytest.approx(2.5)


def test_best_below_one_draws_its_starts_inside_the_bounding_box(monkeypatch, right_triangle):
    seen = {}
    real = K.irls_multi

    def spy(points, p, starts, **kw):
        seen["starts"] = np.array(starts)
        return real(points, p, starts, **kw)

    monkeypatch.setattr(K, "irls_multi", spy)
    shifted = right_triangle + np.array([10.0, 20.0])                                                  # Rahmen (10 bis 14) mal (20 bis 23): der untere Rand ist nicht 0
    K.best_below_one(shifted, 0.5, seed=4, starts=50)
    s = seen["starts"]
    assert s.shape == (50, 2) and (s >= shifted.min(axis=0)).all() and (s <= shifted.max(axis=0)).all()
    assert s[:, 0].max() > 13.0 and s[:, 1].max() > 22.2 and s[:, 0].min() < 11.0                       # die Starts füllen den Rahmen


def test_irls_multi_does_exactly_maxit_steps_and_counts_them(right_triangle):
    starts = np.array([[0.5, 0.5], [3.0, 2.0], [1.0, 2.5]])
    ends, counts = K.irls_multi(right_triangle, 6.0, starts, tol=0.0, maxit=3)                         # tol = 0: keiner scheidet aus
    assert list(counts) == [3, 3, 3]
    for start, end in zip(starts, ends):
        c = start
        for _ in range(3):
            c = K.irls(right_triangle, 6.0, c, maxit=1).center
        assert end == pytest.approx(c, abs=1e-12)
    one_end, one_count = K.irls_multi(right_triangle, 1.5, starts, tol=1e30, maxit=50)                  # Toleranz groß: jeder Start scheidet nach dem ersten Schritt aus
    assert list(one_count) == [1, 1, 1] and one_end == pytest.approx(np.array([K.irls(right_triangle, 1.5, s, maxit=1).center for s in starts]))


def test_newton_does_exactly_maxit_steps_and_each_one_improves(right_triangle):
    start = np.array([5.0, 5.0])
    costs = []
    for maxit in (1, 2, 3):
        r = K.newton_lp(right_triangle, 6.0, c0=start, maxit=maxit)
        assert r.status == "maxit" and r.iterations == maxit
        costs.append(K.lp_cost(right_triangle, r.center, 6.0))
    assert costs[0] < 0.9 * K.lp_cost(right_triangle, start, 6.0) and costs[1] < costs[0] and costs[2] < costs[1]      # schon ein Schritt senkt die Kosten deutlich
    assert K.newton_lp(right_triangle, 6.0, c0=start).iterations > 3
    one = K.newton_lp(right_triangle, 6.0, c0=start, maxit=1).center                                 # zwei Schritte in einem Lauf = zwei Läufe zu je einem Schritt
    assert K.newton_lp(right_triangle, 6.0, c0=start, maxit=2).center == pytest.approx(K.newton_lp(right_triangle, 6.0, c0=one, maxit=1).center, abs=1e-12)
