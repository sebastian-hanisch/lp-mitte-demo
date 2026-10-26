"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, alle Kunden am selben Ort, Gleichstände, Grenzen inklusive."""

import math

import numpy as np
import pytest

import lpm_center as K
import lpm_constants as C
import lpm_evaluation as V
import lpm_regret as RG
import lpm_scenario as S


def test_single_customer_everywhere():
    one = np.array([[2.0, -1.0]])
    assert K.centroid(one) == pytest.approx([2.0, -1.0])
    assert K.newton_lp(one, 3.0).center == pytest.approx([2.0, -1.0]) and K.one_center(one)[1] == 0.0
    assert K.lp_center(one, 1.0) == pytest.approx([2.0, -1.0]) and K.lp_center(one, 0.5) == pytest.approx([2.0, -1.0])
    assert K.solve_1d(np.array([4.0]), 0.5) == (4.0, 4.0) and K.solve_1d(np.array([4.0]), math.inf) == (4.0, 4.0) and K.solve_1d(np.array([4.0]), 1.0) == (4.0, 4.0)
    assert RG.regret_matrix(one) == pytest.approx(np.zeros((3, 3)), abs=1e-12)


def test_exactly_two_customers_every_exponent_gives_the_midpoint(two_points):
    for p in (0.5, 1.0, 1.5, 2.0, 7.0, math.inf):
        c = K.lp_center(two_points, p)
        if p >= 1.0:
            assert c == pytest.approx([3.0, 0.0], abs=1e-7), p
        else:
            assert any(np.allclose(c, q) for q in two_points)      # p < 1: ein Kunde (beide gleich gut)
    assert K.solve_1d(np.array([0.0, 10.0]), 1.0) == (0.0, 10.0) and K.solve_1d(np.array([0.0, 10.0]), 3.0)[0] == pytest.approx(5.0)
    assert K.solve_1d(np.array([0.0, 10.0]), 0.5) == (0.0, 0.0)                    # Gleichstand: der kleinere Wert


def test_all_customers_at_one_place_for_every_solver():
    same = np.array([[1.0, 1.0]] * 5)
    for p in (0.5, 1.0, 1.5, 2.0, 3.0, 8.0, math.inf):
        assert K.lp_center(same, p) == pytest.approx([1.0, 1.0]), p
    assert K.irls(same, 3.0, [1.0, 1.0]).status == "ok" and K.irls(same, 3.0, [2.0, 2.0], relax=0.5).center == pytest.approx([1.0, 1.0], abs=1e-6)
    assert K.newton_lp(same, 4.0, c0=[3.0, -3.0]).center == pytest.approx([1.0, 1.0], abs=1e-6)


def test_collinear_customers_have_their_center_on_the_line():
    line = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [9.0, 9.0]])
    for p in (1.5, 2.0, 4.0):
        c = K.lp_center(line, p)
        assert c[0] == pytest.approx(c[1], abs=1e-8)
    assert K.lp_center(line, 4.0)[0] == pytest.approx(K.solve_1d(np.array([0.0, 1.0, 2.0, 9.0]) * math.sqrt(2), 4.0)[0] / math.sqrt(2), abs=1e-7)


def test_the_exponent_one_is_the_boundary_between_the_two_regimes():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [5.0, 5.0]])
    near = K.lp_center(pts, 1.0 + 1e-6)
    assert near == pytest.approx(K.lp_center(pts, 1.0), abs=2e-3)                      # stetig an p = 1 von oben
    x = np.array([0.0, 1.0, 4.0, 10.0])
    assert K.solve_1d(x, 1.0) == (1.0, 4.0) and K.solve_1d(x, 1.0 + 1e-9)[0] == pytest.approx(K.solve_1d(x, 1.0 + 1e-9)[1]) and 1.0 <= K.solve_1d(x, 1.0 + 1e-9)[0] <= 4.0
    assert K.solve_1d(x, 1.0 - 1e-9)[0] in (0.0, 1.0, 4.0, 10.0)                        # darunter ein Datenpunkt


def test_group_size_and_layout_bounds_are_inclusive():
    assert S.group_size(0.5, 5) == 3 and S.group_size(0.95, 5) == 4 and S.group_size(0.7, 2) == 1
    pts = S.make_points("Zwei Häufungen", C.N_MIN, 3, share=C.SHARE_MAX / 100.0)
    assert pts.shape == (C.N_MIN, 2) and (pts[:, 0] > 3).sum() >= 1                    # die kleine Häufung ist auch bei 95 % und n = 5 nicht leer


def test_analysis_at_the_corners_of_the_regler():
    smallest = V.analyse(V.Settings(n=C.N_MIN, p=C.P_OPTIONS[0], outpost=C.OUTPOST_MIN))
    largest = V.analyse(V.Settings(layout="Außenposten", n=C.N_MAX, p=C.P_FINITE_MAX, outpost=C.OUTPOST_MAX, solver=C.SOLVERS[2]))
    for a in (smallest, largest):
        assert np.isfinite(a.regrets).all() and np.isfinite(a.reference).all() and a.nearest >= 0.0
    assert largest.solve.status == "ok" and largest.solver_gap < 1e-4
    infinite = V.analyse(V.Settings(p=math.inf, n=C.N_MIN))
    assert infinite.regrets[2] == pytest.approx(0.0, abs=1e-9) and infinite.reference == pytest.approx(infinite.one_center)
    degenerate = V.analyse(V.Settings(layout="Außenposten", outpost=0.0, n=C.N_MIN, p=1.0))
    assert np.isfinite(degenerate.regrets).all()


def test_the_fmt_p_of_every_option_roundtrips_through_the_comma():
    for p in C.P_OPTIONS:
        text = V.fmt_p(p)
        assert (text == "∞") == (p == math.inf) and (p == math.inf or float(text.replace(",", ".")) == p)
