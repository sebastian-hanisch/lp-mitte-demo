"""Regret-Einheiten an der Handinstanz (rechtwinkliges Dreieck): jede Zahl ist in conftest.py von Hand gerechnet."""

import math

import numpy as np
import pytest

import lpm_regret as RG
from lpm_center import lp_cost


def test_regret_percent_and_the_zero_denominator():
    assert RG.regret_percent(11.0, 10.0) == pytest.approx(10.0) and RG.regret_percent(10.0, 10.0) == 0.0
    assert RG.regret_percent(0.0, 0.0) == 0.0 and RG.regret_percent(3.0, 0.0) == 0.0


def test_optimal_centers_of_the_right_triangle(right_triangle):
    oc = RG.optimal_centers(right_triangle)
    assert set(oc) == {1.0, 2.0, math.inf}
    assert oc[2.0] == pytest.approx([4 / 3, 1.0]) and oc[math.inf] == pytest.approx([2.0, 1.5])


def test_regret_matrix_of_the_right_triangle_by_hand(right_triangle):
    """Summe: Schwerpunkt 6,9184 und 1-Center 7,5 gegen das Optimum. Quadratsumme: 1-Center 18,75 gegen 16,667 (+12,5 %). Längster Weg: Schwerpunkt 2,848 gegen 2,5."""
    m = RG.regret_matrix(right_triangle)
    g = 5 / 3 + math.sqrt(73) / 3 + math.sqrt(52) / 3
    opt_sum = lp_cost(right_triangle, RG.optimal_centers(right_triangle)[1.0], 1)
    assert opt_sum < g
    assert m[1, 0] == pytest.approx(100 * (g / opt_sum - 1)) and m[2, 0] == pytest.approx(100 * (7.5 / opt_sum - 1))
    assert m[2, 1] == pytest.approx(100 * (18.75 / (150 / 9) - 1)) == pytest.approx(12.5)
    assert m[1, 2] == pytest.approx(100 * (math.sqrt(73) / 3 / 2.5 - 1))
    assert m[1, 1] == m[0, 0] == m[2, 2] == pytest.approx(0.0, abs=1e-9)
    assert m[0, 1] > 0 and m[0, 2] > 0 and (m >= -1e-9).all()


def test_regret_row_matches_the_matrix_rows(right_triangle):
    oc = RG.optimal_centers(right_triangle)
    m = RG.regret_matrix(right_triangle)
    assert RG.regret_row(right_triangle, oc[1.0], oc) == pytest.approx(m[0])
    assert RG.regret_row(right_triangle, oc[math.inf], oc) == pytest.approx(m[2])


def test_all_customers_at_one_place_have_no_regret():
    same = np.array([[1.0, 1.0]] * 4)
    assert RG.regret_matrix(same) == pytest.approx(np.zeros((3, 3)), abs=1e-9)


def test_compromise_regrets_are_the_row_maxima_and_the_grid_endpoints_agree(right_triangle):
    m = RG.regret_matrix(right_triangle)
    r = RG.compromise_regrets(right_triangle, (1.0, 2.0, math.inf))
    assert r == pytest.approx(m.max(axis=1))
    mid = RG.compromise_regrets(right_triangle, (1.0, 3.0, 2.0, math.inf))
    assert mid[0] == pytest.approx(r[0]) and mid[3] == pytest.approx(r[2]) and mid[2] == pytest.approx(r[1])
    assert mid[1] < min(r)                                              # ein p dazwischen ist hier besser als jedes der drei Enden


def test_empty_zone(right_triangle):
    assert RG.empty_zone(right_triangle, np.array([2.0, 1.5]), 2.0) and not RG.empty_zone(right_triangle, np.array([2.0, 1.5]), 2.5)
    assert not RG.empty_zone(right_triangle, np.array([0.0, 0.0]), 0.0)
    assert RG.empty_zone(right_triangle, np.array([-3.0, 0.0]), 2.0) and not RG.empty_zone(right_triangle, np.array([-3.0, 0.0]), 3.0)      # nächster Kunde (0; 0) im Abstand 3
