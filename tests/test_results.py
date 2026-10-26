"""Die vorgerechneten Messreihen (data/lpm_results.json): vollständig, aktuell gegenüber den Quelldateien und in sich stimmig."""

import json
import math

import numpy as np
import pytest

import lpm_constants as C
import lpm_experiments as X
import lpm_results as R


def test_results_file_is_current_with_the_generator_sources():
    """Schlägt fehl, sobald sich lpm_rng, lpm_scenario, lpm_geomed, lpm_center, lpm_regret, lpm_experiments oder tools/make_results.py ändern:
    dann `python tools/make_results.py --part <m1a|m1b|m2|m4|m5|m6a|m6b|m7>` und `--merge` neu laufen lassen."""
    assert R.meta()["erzeuger_hash"] == X.source_hash()
    assert R.meta()["quellen"] == list(X.SOURCES) and R.meta()["seeds"] == list(C.SEEDS)


def test_results_file_is_valid_utf8_json_with_all_parts_and_no_infinity_literals():
    text = R.PATH.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"meta", "m1a", "m1b", "m2", "m4", "m5", "m6a", "m6b", "m7"}
    assert "Zwei Häufungen" in text and "Infinity" not in text and "NaN" not in text      # Umlaute im Klartext, gültiges JSON


def test_every_part_of_the_generator_is_in_the_file():
    assert set(X.PARTS) == set(R.load()) - {"meta"}


def test_infinity_is_stored_as_text_and_read_back():
    assert R.p_of("inf") == math.inf and R.p_of(2.0) == 2.0 and X.unjp(X.jp(math.inf)) == math.inf and X.jp(3.0) == 3.0
    assert R.example_row(math.inf)["lo"] == 20.5 and R.example_row(1.0) == {"p": 1.0, "lo": 5.0, "hi": 6.0}


def test_axis_and_plane_cells_are_complete():
    assert len(R.axis_cells()) == 4 * len(X.MONOTONE_N_AXIS) and len(R.plane_cells()) == len(C.LAYOUTS) * len(X.MONOTONE_N_PLANE)
    for c in R.axis_cells():
        assert 0.0 <= c["strict"][0] <= 1.0 and c["strict_to_8"][0] >= c["strict"][0] - 1e-12 and c["runs_per_seed"] == 125
    assert R.axis_cell("Normal", 51)["n"] == 51 and R.plane_cell("Außenposten", 30)["layout"] == "Außenposten"
    assert [r["p"] for r in R.approach_cell("Normal")["rows"]] == list(X.APPROACH_P) and R.approach_row("Normal", 8.0)["p"] == 8.0
    assert [c["layout"] for c in R.load()["m1b"]["approach"]] == list(C.LAYOUTS)


def test_regret_raster_and_outposts_are_complete():
    assert len(R.regret_cells()) == len(C.LAYOUTS) * len(X.N_RASTER)
    for c in R.regret_cells():
        assert all(abs(c["mean"][i][i]) < 1e-9 for i in range(3)) and min(min(r) for r in c["mean"]) > -1e-9
    assert [r["a"] for r in R.outpost_rows()] == list(C.OUTPOST_RASTER) and R.outpost_row(6.0)["n"] == 31


def test_compromise_cells_are_complete_and_the_best_is_the_minimum():
    assert len(R.compromise_cells()) == len(C.LAYOUTS) * len(X.COMPROMISE_N)
    for c in R.compromise_cells():
        assert [X.unjp(p) for p in c["grid"]] == list(C.GRID_P)
        assert c["best_value"] == min(c["mean"]) and c["mean"][list(c["grid"]).index(c["best_p"])] == c["best_value"]
        assert c["free_choice"] <= c["best_value"] + 1e-12            # je Instanz frei wählen ist nie schlechter als ein festes p
    assert R.compromise_cell("Außenposten", 30)["layout"] == "Außenposten"


def test_other_parts_have_their_rows():
    assert len(R.empty_plane_cells()) == 8 and R.empty_plane_value(0.6, 10.0, 1.0) == 0.0 and len(R.empty_axis_cells()) == 5
    assert R.empty_axis_value(0.55, 2.0) == pytest.approx(R.empty_axis_value(0.55, 2.0))
    assert [c["s_big"] for c in R.density_cells()] == [0.5, 1.0, 2.0, 3.0, 4.0] and R.density_value(3.0, 0.5) > 0.5
    assert [r["p"] for r in R.start_rows()] == list(X.START_P) and R.start_row(0.5)["p"] == 0.5
    assert len(R.stair_rows()) == 12 and R.stair_row("Normal", 0.1, 0.1)["lo"] == 0.1
    assert len(R.plane_data_cells()) == 4 and 0.0 <= R.plane_data_value("Normal", 5, 0.9) <= 1.0
    assert [c["spread"] for c in R.plane_density_cells()] == [1.0, 2.0, 3.0] and R.plane_density_value(3.0, 0.4) > 0.5
    assert [r["p"] for r in R.plane_start_rows()] == [0.7, 0.5, 0.2] and R.plane_start_row(0.2)["p"] == 0.2
    assert [c["layout"] for c in R.irls_cells()] == ["Normal", "Zwei Häufungen"] and R.irls_row("Normal", 4.0)["p"] == 4.0


def test_stored_cells_are_reproduced_by_the_live_functions():
    """Das Nachmessen in der App liefert dieselben Zahlen wie die vorgerechnete Zelle (gleiche Seeds)."""
    cell = R.regret_cell("Außenposten", 10)
    again = X.regret_cell("Außenposten", 10, cell["runs_per_seed"])
    assert np.asarray(again["mean"]) == pytest.approx(np.asarray(cell["mean"])) and np.asarray(again["std"]) == pytest.approx(np.asarray(cell["std"]))
    c4 = R.compromise_cell("Quadrat", 10)
    live = X.compromise_cell("Quadrat", 10, c4["runs_per_seed"])
    assert live["best_p"] == c4["best_p"] and live["mean"] == pytest.approx(c4["mean"])
    e = R.empty_plane_cell(0.6, 10.0)
    assert X.empty_plane_cell(0.6, 10.0, 100, e["runs_per_seed"])["rows"] == e["rows"]
    d = R.density_cells()[3]
    assert X.density_axis_cell(3.0, 200, d["runs_per_seed"])["rows"] == d["rows"]
