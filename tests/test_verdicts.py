"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen oder eine Analyse, nicht Seeds."""

import math
from dataclasses import replace

import numpy as np
import pytest

import lpm_center as K
import lpm_constants as C
import lpm_evaluation as V


def _analysis(**settings):
    return V.analyse(V.Settings(**{**C._BASE, **settings}))


def _with_regrets(a, values):
    return replace(a, regrets=np.array(values, dtype=float))


def test_formatting_helpers():
    assert V.de(3.14159, 2) == "3,14" and V.pct(0.256, 1) == "25,6" and V.pct(0.5, 0) == "50"
    assert V.fmt_p(math.inf) == "∞" and V.fmt_p(0.5) == "0,5" and V.fmt_p(1.0) == "1" and V.fmt_p(1.25) == "1,25" and V.fmt_p(64.0) == "64"
    assert V.unjp("inf") == math.inf and V.unjp(3) == 3.0


def test_verdict_regret_thresholds_and_numbers():
    a = _analysis(p=3.0)
    kind, text = V.verdict_regret(_with_regrets(a, [0.2, 0.9, 0.5]))
    assert kind == "success" and "0,2 % mehr Summe der Wege" in text and "0,9 % mehr Quadratsumme" in text and "0,5 % mehr längsten Weg" in text
    assert "Nach Ihrem Kriterium „Summe der Wege (q = 1)“ liegt die Mitte für p = 3 **0,2 %** über dem Optimum (der besten Mitte dafür, p = 1)" in text    # Standardkriterium q = 1
    assert "am teuersten wird sie nach dem Kriterium „Quadratsumme“" in text                                  # das Kriterium mit dem größten Regret
    other = V.verdict_regret(_with_regrets(replace(a, settings=replace(a.settings, criterion=C.CRITERIA[2])), [0.2, 0.9, 0.5]))[1]
    assert "Nach Ihrem Kriterium „längster Weg (q = ∞)“ liegt die Mitte für p = 3 **0,5 %** über dem Optimum (der besten Mitte dafür, p = ∞)" in other
    assert "**0,9 %**" in V.verdict_regret(_with_regrets(replace(a, settings=replace(a.settings, criterion=C.CRITERIA[1])), [0.2, 0.9, 0.5]))[1]
    assert V.verdict_regret(_with_regrets(a, [0.0, 1.0, 0.0]))[0] == "info"                  # genau 1 %: nicht mehr „success“
    assert V.verdict_regret(_with_regrets(a, [9.9, 0.0, 0.0]))[0] == "info"
    kind, text = V.verdict_regret(_with_regrets(a, [0.0, 0.0, 10.0]))                       # genau 10 %: „warning“
    assert kind == "warning" and "„längster Weg“" in text and "„Summe der Wege“" not in text
    assert "„Summe der Wege“" in V.verdict_regret(_with_regrets(a, [7.0, 1.0, 2.0]))[1]


def test_verdict_regret_below_one_names_where_the_minimum_lies():
    a = replace(_analysis(p=0.5), reference_on_data=True, nearest=0.0, regrets=np.array([1.0, 2.0, 3.0]))
    kind, text = V.verdict_regret(a)
    assert kind == "info" and "p = 0,5 < 1" in text and "**auf einem Kunden**" in text and "0,00 km" in text and "1,0 % mehr Summe der Wege" in text and "3,0 % mehr längsten Weg" in text
    inner = V.verdict_regret(replace(a, reference_on_data=False, nearest=0.31))[1]
    assert "im Innern zwischen den Kunden" in inner and "0,31 km" in inner


def test_verdict_solver_branches():
    assert "kleinste umschließende Kreis" in V.verdict_solver(_analysis(p=math.inf))[1]
    assert V.verdict_solver(_analysis(p=2.0)) == ("info", "Für p = 2 ist die Mitte der Schwerpunkt: eine Formel, keine Iteration.")
    one = V.verdict_solver(_analysis(p=1.0))
    assert one[0] == "info" and "vom Vardi–Zhang-Löser" not in one[1] and "Vardi–Zhang-Löser aus Stück 5" in one[1]
    plain = _analysis(p=6.0, solver=C.SOLVERS[1])
    kind, text = V.verdict_solver(plain)
    assert kind == "error" and "Grenze von 3000 Schritten" in text and f"{V.de(plain.solver_gap, 2)} km" in text
    ok_kind, ok_text = V.verdict_solver(_analysis(p=3.0, solver=C.SOLVERS[0]))
    assert ok_kind == "success" and "Newton konvergiert nach" in ok_text and "Schritten, Abstand zur Mitte" in ok_text
    base = _analysis(p=3.0, solver=C.SOLVERS[0])
    assert V.verdict_solver(replace(base, solve=K.Solve(base.center, 7, "stillstand", None)))[0] == "warning"
    assert V.verdict_solver(replace(base, solve=K.Solve(base.center, 7, "divergent", None)))[0] == "error"
    below = _analysis(p=0.4, layout="Zwei Häufungen", n=100, share=60, sep=8.0, spread=3.0, seed=8)
    assert V.verdict_solver(below)[0] == "warning" and "lokales Minimum" in V.verdict_solver(below)[1]
    lucky = replace(below, solver_gap=0.0)
    assert V.verdict_solver(lucky)[0] == "success" and f"nach {lucky.solve.iterations} Schritten" in V.verdict_solver(lucky)[1]


def test_verdict_e1_example():
    rows = [{"p": 1.0, "lo": 5.0, "hi": 6.0}, {"p": 8.0, "lo": 19.0, "hi": 19.0}, {"p": 32.0, "lo": 20.0, "hi": 20.0}, {"p": "inf", "lo": 20.5, "hi": 20.5}]
    kind, text = V.verdict_e1_example(rows)
    assert kind == "info" and "Intervall 5 bis 6" in text and "für p = 8 liegt sie bei 19,00" in text and "p = 32 bei 20,00" in text and "p = ∞ bei 20,50" in text and "fehlen noch 0,50 Einheiten" in text


def test_verdict_e1_axis_and_plane():
    def axis(n, strict, to8, back_mean, back_max):
        return {"n": n, "strict": (strict, 0.0), "strict_to_8": (to8, 0.0), "back_mean": back_mean, "back_max": back_max}
    kind, text = V.verdict_e1_axis(axis(51, 0.30, 0.41, 0.0036, 0.047), axis(51, 0.98, 0.99, 0.0, 0.005), axis(51, 0.47, 0.92, 0.0014, 0.019))
    assert kind == "warning" and "Bei 51 normalverteilten" in text and "nur in 30 %" in text and "bei zwei Häufungen in 47 % (bis p = 8: 92 %)" in text and "lognormalen Werten in 98 %" in text
    assert "im Mittel 0,36 % der Spannweite" in text and "höchstens 4,7 %" in text
    plane = {"n": 30, "sum_strict": (0.53, 0.0), "long_strict": (0.51, 0.0), "sum_back_max_pct": 1.65, "long_back_max_pct": 4.23}
    out = {"n": 30, "sum_strict": (0.91, 0.0), "long_strict": (0.88, 0.0), "sum_back_max_pct": 0.79, "long_back_max_pct": 7.5}
    kind, text = V.verdict_e1_plane(plane, out)
    assert kind == "warning" and "Bei 30 normalverteilten Kunden wächst die Summe der Wege nur in 53 %" in text and "nur in 51 % ohne Rückschritt" in text and "sind es 91 % und 88 %" in text
    assert "größte Rückschritt beträgt 7,5 %" in text                       # das Maximum über beide Layouts und beide Größen
    assert "beträgt 4,2 %" in V.verdict_e1_plane(plane, {**out, "long_back_max_pct": 1.0})[1]


def test_verdict_e1_approach_takes_min_and_max_over_the_layouts():
    def cell(layout, d8, d32):
        return {"layout": layout, "n": 30, "rows": [{"p": 8.0, "dist_radius": d8}, {"p": 32.0, "dist_radius": d32}]}
    kind, text = V.verdict_e1_approach([cell("a", 0.057, 0.0231), cell("b", 0.117, 0.0261), cell("c", 0.076, 0.0245)])
    assert kind == "info" and "Bei p = 8 liegt c(p) noch 6 bis 12 %" in text and "bei p = 32 noch 2,3 bis 2,6 %" in text and "n = 30" in text


def test_verdict_e2_e3():
    def cell(n, m):
        return {"n": n, "mean": m}
    out = [[0, 1, 59], [1, 0, 55.1], [137, 214.2, 0]]
    nor = [[0, 1, 15.8], [0.5, 0, 13.4], [8.3, 13.1, 0]]
    kind, text = V.verdict_e2(cell(30, out), cell(30, nor))
    assert kind == "warning" and "(n = 30) zahlt" in text and "**214 %**" in text and "längsten Weg bewertet, 55 %" in text and "(Normal, n = 30) sind es 13 % und 13 %" in text
    rows = [{"a": a, "mean": [[0, 0, v], [0, 0, 0], [0, 0, 0]]} for a, v in ((2.0, 14.1), (3.0, 31.3), (6.0, 59.6))]
    kind, text = V.verdict_e3(rows)
    assert kind == "info" and "a = 2 noch bei 14,1 %" in text and "a = 3 bei 31,3 %" in text and "a = 6 bei 59,6 %" in text


def test_verdict_e4_names_the_best_p_per_layout_and_the_regret_of_p_two():
    grid = [1.0, 2.0, 3.0, "inf"]

    def cell(layout, best_p, best_value, p2):
        return {"layout": layout, "n": 30, "grid": grid, "best_p": best_p, "best_value": best_value, "mean": [50.0, p2, best_value, 70.0]}
    cells = [cell("Normal", 3.0, 6.8, 12.9), cell("Quadrat", "inf", 3.9, 8.4), cell("Außenposten", 3.0, 42.2, 54.8), cell("Zwei Häufungen", 3.0, 14.8, 21.1)]
    kind, text = V.verdict_e4(cells)
    assert kind == "info" and "Normal p = 3 (6,8 %)" in text and "Quadrat p = ∞ (3,9 %)" in text and "Außenposten p = 3 (42,2 %)" in text and "n = 30" in text
    assert "Regret liegt bei 12,9 % (Normal) bzw. 54,8 % (Außenposten)" in text


def test_verdict_e5():
    plane = {"share": 0.6, "sep": 10.0, "rows": [{"p": 1.0, "mean": 0.0}, {"p": 2.0, "mean": 0.81}, {"p": "inf", "mean": 0.98}]}
    axis = {"share": 0.55, "rows": [{"p": 2.0, "mean": 0.6}, {"p": 4.0, "mean": 0.74}]}
    kind, text = V.verdict_e5(plane, axis)
    assert kind == "warning" and "Abstand 10 km mit 60 %" in text and "(p = 2) in 81 %" in text and "(p = 1) in 0 %" in text and "1-Center in 98 %" in text
    assert "(Anteil 55 %) sind es für p = 2 60 % und für p = 4 74 %" in text


def test_verdict_e6():
    cell = {"s_big": 3.0, "rows": [{"p": 1.0, "mean": 0.0}, {"p": 0.7, "mean": 0.03}, {"p": 0.5, "mean": 0.84}, {"p": 0.3, "mean": 1.0}]}
    kind, text = V.verdict_e6_density(cell)
    assert kind == "success" and "mit 3 um 0" in text and "bei p = 1, in 3 % bei p = 0,7" in text and "in 84 % bei p = 0,5 und in 100 % bei p = 0,3" in text
    start = [{"p": 0.7, "B": (0.61, 0), "A": (1.0, 0), "distinct": (15.0, 0), "starts": 200}, {"p": 0.5, "B": (0.07, 0), "A": (0.68, 0), "distinct": (58.0, 0), "starts": 200},
             {"p": 0.2, "B": (0.03, 0), "A": (0.64, 0), "distinct": (84.0, 0), "starts": 200}]
    kind, text = V.verdict_e6_start(start)
    assert kind == "warning" and "p = 0,7 enden 61 %" in text and "p = 0,5 7 %" in text and "p = 0,2 3 %" in text and "enden 68 % bzw. 64 % (p = 0,5 bzw. 0,2)" in text and "im Mittel 84 verschiedene Enden bei 200 Starts" in text
    cells = [{"rows": [{"p": 0.9, "mean": 0.61}, {"p": 0.7, "mean": 0.99}, {"p": 0.5, "mean": 1.0}]}, {"rows": [{"p": 0.9, "mean": 0.14}, {"p": 0.7, "mean": 0.62}, {"p": 0.5, "mean": 0.98}]}]
    kind, text = V.verdict_e6_plane(cells)
    assert kind == "info" and "für p = 0,5 in 98 bis 100 %" in text and "für p = 0,7 in 62 bis 99 %" in text and "für p = 0,9 nur in 14 bis 61 %" in text


def test_verdict_e7():
    def row(p, plain, damped, newton):
        return {"p": p, "plain": plain, "damped": damped, "newton_median_its": newton}
    never = {"converged": 0.0, "rises": 1.0, "median_its": None}
    normal = {"maxit": 5000, "rows": [row(4.0, never, {"converged": 1.0, "median_its": 23.5}, 3.0), row(8.0, never, {"converged": 1.0, "median_its": 42.5}, 4.0)]}
    two = {"rows": [row(3.0, {"converged": 1.0, "rises": 0.06, "median_its": 198.0}, {"converged": 1.0, "median_its": 25.0}, 4.0)]}
    kind, text = V.verdict_e7(normal, two)
    assert kind == "error" and "bei p = 4 in 0 % der Läufe (Normal; Grenze 5000 Schritte)" in text and "steigt in 100 %" in text and "bei p = 3 mit zwei Häufungen steigt er in 6 %" in text
    assert "im Median 198 Schritte" in text and "in 100 % der Läufe bei p = 8 (Median 42 Schritte; Newton: 4)" in text
