"""Die Hilfsfunktionen der Messreihen (lpm_experiments.py) an kleinen, von Hand prüfbaren Fällen und mit wenigen Instanzen."""

import math

import numpy as np
import pytest

import lpm_center as K
import lpm_constants as C
import lpm_experiments as X
from lpm_scenario import Draws, axis_batch, cell_seed


def test_reversal_counts_only_steps_against_the_direction():
    assert X.reversal([0, 1, 2, 3], +1, 3.0) == (True, 0.0)
    strict, back = X.reversal([0, 2, 1, 3], +1, 3.0)
    assert not strict and back == pytest.approx(1 / 3)
    assert X.reversal([3, 2, 3.5, 0], -1, 2.0) == (False, pytest.approx(0.75))
    assert X.reversal([5, 5, 5], +1, 1.0) == (True, 0.0)                         # Stillstand ist kein Rückschritt
    assert X.reversal([0, 1 - 1e-12, 1 - 2e-12], +1, 1.0)[0] is True              # genau die Toleranz 1e-12 · scale gilt noch als strikt


def test_increases_has_a_relative_tolerance():
    assert not X.increases([3.0, 2.0, 1.0]) and X.increases([3.0, 2.0, 2.5]) and not X.increases([1.0, 1.0, 1.0])
    assert not X.increases([1.0, 1.0 + 1e-13]) and X.increases([1.0, 1.0 + 1e-10])


def test_number_example_by_hand():
    rows = {X.unjp(r["p"]): r for r in X.number_example()["rows"]}
    assert (rows[1.0]["lo"], rows[1.0]["hi"]) == (5.0, 6.0)
    assert rows[2.0]["lo"] == pytest.approx((45 + 40) / 10) and rows[math.inf]["lo"] == pytest.approx((1 + 40) / 2)
    assert rows[128.0]["lo"] < 20.5 < rows[128.0]["lo"] + 0.01 and rows[1.5]["lo"] < rows[3.0]["lo"] < rows[8.0]["lo"]
    assert X.number_example()["values"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 40]


def test_small_helpers():
    assert X.mean_std([1.0, 3.0]) == (2.0, 1.0) and X.jp(math.inf) == "inf" and X.jp(2) == 2.0
    assert X.live_runs(10) == C.LIVE_BUDGET_INSTANCES == 400 and X.live_runs(300) == 133 and X.live_runs(5000) == 20 and X.live_runs(100, per_instance_cost=5.0) == 80
    assert len(X.source_hash()) == 16 and X.source_hash() == X.source_hash()


def test_axis_monotone_cell_matches_a_direct_recomputation():
    cell = X.axis_monotone_cell("Zwei Häufungen", 21, 6, seeds=(11,))
    strict = 0
    for x in axis_batch(Draws(cell_seed(11, "axis", "Zwei Häufungen", 21)), "Zwei Häufungen", 6, 21):
        path = np.array([K.solve_1d(x, p)[0] for p in X.AXIS_P])
        d = 1 if path[-1] >= path[0] else -1
        strict += bool(np.all(d * np.diff(path) >= -1e-12 * (x.max() - x.min())))
    assert cell["strict"] == (strict / 6, 0.0) and cell["runs_per_seed"] == 6 and 0.0 <= cell["strict_to_8"][0] <= 1.0


def test_plane_monotone_and_approach_cells_have_the_documented_shape():
    cell = X.plane_monotone_cell("Normal", 20, 3, seeds=(11, 12))
    assert 0.0 <= cell["sum_strict"][0] <= 1.0 and cell["sum_back_max_pct"] >= 0.0 and cell["layout"] == "Normal" and cell["n"] == 20
    ap = X.approach_cell("Außenposten", 30, 3, seeds=(11,))
    rows = {r["p"]: r for r in ap["rows"]}
    assert list(rows) == list(X.APPROACH_P)
    assert rows[4.0]["dist_radius"] > rows[16.0]["dist_radius"] > rows[128.0]["dist_radius"] and rows[4.0]["overshoot_pct"] > rows[64.0]["overshoot_pct"] > 0.0
    assert rows[128.0]["sum_excess_pct"] > rows[4.0]["sum_excess_pct"] > 0.0           # Außenposten: die Summe wächst mit p, der längste Weg fällt


def test_regret_and_compromise_cells():
    cell = X.regret_cell("Außenposten", 10, 4, seeds=(11, 12))
    m = np.array(cell["mean"])
    assert m.shape == (3, 3) and np.diag(m) == pytest.approx([0, 0, 0], abs=1e-9) and m[2, 1] > m[1, 2]
    comp = X.compromise_cell("Normal", 10, 3, seeds=(11,))
    assert [X.unjp(p) for p in comp["grid"]] == list(C.GRID_P) and comp["best_value"] == min(comp["mean"]) and comp["free_choice"] <= comp["best_value"] + 1e-12
    sub = X.compromise_cell("Normal", 10, 3, seeds=(11,), grid=(1.0, 2.0))
    assert len(sub["mean"]) == 2 and sub["mean"][0] == pytest.approx(comp["mean"][0])


def test_empty_zone_cells():
    plane = X.empty_plane_cell(0.6, 10.0, 100, 4, seeds=(11,))
    assert [X.unjp(r["p"]) for r in plane["rows"]] == [1.0, 2.0, math.inf] and plane["rows"][0]["mean"] == 0.0
    axis = X.empty_axis_cell(0.5, 100, 4, (0.5, 2.0), seeds=(11,))
    assert axis["rows"][0]["mean"] == 0.0 and [r["p"] for r in axis["rows"]] == [0.5, 2.0]            # p < 1 liegt auf einem Wert


def test_below_one_cells():
    dens = X.density_axis_cell(0.5, 100, 4, seeds=(11,))
    assert all(r["mean"] == 0.0 for r in dens["rows"])                                         # gleiche Dichte: die Mehrheit gewinnt für jedes p
    strong = X.density_axis_cell(4.0, 100, 4, seeds=(11,))
    assert {r["p"]: r["mean"] for r in strong["rows"]}[0.3] == 1.0
    st = X.axis_start_study(2.0, 1, n=60, starts=20, seeds=(11,))
    assert st["B"][0] == 1.0 and st["distinct"][0] == 1.0 and st["D"][0] == 0.0 and st["conv"][0] == 1.0          # p > 1: jeder Start endet im selben Punkt, nicht auf einem Wert
    low = X.axis_start_study(0.3, 1, n=60, starts=20, seeds=(11,))
    assert low["D"][0] == 1.0 and low["distinct"][0] > 3
    plane = X.below_one_plane_cell("Normal", 8, 3, seeds=(11,))
    assert {r["p"]: r["mean"] for r in plane["rows"]}[0.1] == 1.0
    stairs_low = X.staircase_cell("Normal", 31, 3, 0.1, 0.9, 0.02, seeds=(11,))["largest_step"][0]
    stairs_high = X.staircase_cell("Normal", 31, 3, 1.1, 3.0, 0.02, seeds=(11,))["largest_step"][0]
    assert stairs_low > 5 * stairs_high
    dplane = X.density_plane_cell(3.0, 60, 3, seeds=(11,))
    assert {r["p"]: r["mean"] for r in dplane["rows"]}[0.2] >= {r["p"]: r["mean"] for r in dplane["rows"]}[2.0]
    pst = X.plane_start_study(0.5, 1, n=40, starts=10, seeds=(11,))
    assert 0.0 <= pst["B"][0] <= 1.0 and 0.0 <= pst["A"][0] <= 1.0


def test_irls_instance_and_cell():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [5.0, 5.0], [1.0, 2.0]])
    r = X.irls_instance(pts, 3.0, 2000)
    assert r["newton_status"] == "ok" and r["plain"]["status"] == "ok" and r["damped"]["status"] == "ok"
    assert r["damped"]["its"] < r["plain"]["its"] and r["plain"]["error"] < 1e-5 and r["damped"]["error"] < 1e-5
    hard = X.irls_instance(pts, 8.0, 500)
    assert hard["plain"]["status"] == "maxit" and hard["plain"]["rises"] and hard["damped"]["status"] == "ok" and not hard["damped"]["rises"]
    cell = X.irls_cell("Normal", 20, 2, seeds=(11,), maxit=500)
    rows = {row["p"]: row for row in cell["rows"]}
    assert list(rows) == list(X.IRLS_P) and rows[6.0]["plain"]["converged"] == 0.0 and rows[6.0]["plain"]["median_its"] is None
    assert rows[6.0]["damped"]["converged"] == 1.0 and rows[1.5]["plain"]["median_its"] == rows[1.5]["damped"]["median_its"]     # für p ≤ 2 gibt es keine Dämpfung


def test_parts_registry_matches_the_documented_parts():
    assert set(X.PARTS) == {"m1a", "m1b", "m2", "m4", "m5", "m6a", "m6b", "m7"}
