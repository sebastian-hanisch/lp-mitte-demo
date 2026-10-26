"""Plotly-Figuren: bauen ohne Fehler, Achsen fest, die erwarteten Spuren sind da."""

import math

import numpy as np
import pytest

import lpm_constants as C
import lpm_evaluation as V
import lpm_experiments as X
import lpm_results as R
import lpm_visualization as VIS


def _fixed(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_map_has_every_mark_and_the_circle():
    a = V.analyse(V.Settings(layout="Außenposten", p=3.0))
    fig = VIS.build_map(a)
    names = [t.name for t in fig.data if t.name]
    assert "Kunden" in names and "Kleinster umschließender Kreis" in names and "Schwerpunkt (p = 2)" in names and "Geometrischer Median (p = 1)" in names and "1-Center (p = ∞)" in names
    assert "Gewählte Mitte (p = 3)" in names and "Ergebnis des Lösers" not in names and _fixed(fig) and fig.layout.yaxis.scaleanchor == "x"
    circle = next(t for t in fig.data if t.name == "Kleinster umschließender Kreis")
    assert np.hypot(np.array(circle.x) - a.one_center[0], np.array(circle.y) - a.one_center[1]) == pytest.approx(a.one_radius)
    labels_on = [t for t in fig.data if t.name == "Kunden"][0].text
    assert labels_on is not None and len(labels_on) == 30
    big = VIS.build_map(V.analyse(V.Settings(n=100)))
    assert [t for t in big.data if t.name == "Kunden"][0].text is None                      # ab 31 Kunden keine Nummern


def test_map_shows_the_solver_result_only_when_it_differs():
    hanging = V.analyse(V.Settings(p=6.0, solver=C.SOLVERS[1]))
    assert "Ergebnis des Lösers" in [t.name for t in VIS.build_map(hanging).data]
    assert "Gewählte Mitte (p = ∞)" in [t.name for t in VIS.build_map(V.analyse(V.Settings(p=math.inf))).data]


def test_axis_path_figure():
    xs = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 40])
    ps = (0.5, 1.0, 2.0, 4.0, math.inf)
    fig = VIS.build_axis_path(xs, ps, [(1.0, 1.0), (5.0, 6.0), (8.5, 8.5), (16.0, 16.0), (20.5, 20.5)])
    names = [t.name for t in fig.data]
    assert "Median-Intervall (p = 1)" in names and _fixed(fig) and list(fig.data[1].x) == [1.0, 5.5, 8.5, 16.0] and list(fig.data[1].y) == [0.5, 1.0, 2.0, 4.0]
    no_interval = VIS.build_axis_path(xs, (1.0, 2.0), [(5.0, 5.0), (8.5, 8.5)])
    assert "Median-Intervall (p = 1)" not in [t.name for t in no_interval.data]


def test_experiment_figures_build_from_the_stored_results():
    figs = [VIS.build_example_path(R.example()["rows"]), VIS.build_approach_chart(R.load()["m1b"]["approach"]), VIS.build_regret_heatmap(R.regret_cell("Normal", 30)["mean"], "t"),
            VIS.build_regret_by_n(R.regret_cells(), 1, 2, "t"), VIS.build_outpost_chart(R.outpost_rows()), VIS.build_compromise_chart(R.compromise_cell("Normal", 30)),
            VIS.build_compromise_chart(R.compromise_cell("Normal", 30), [R.compromise_cell("Quadrat", 30)]), VIS.build_empty_chart([c for c in R.empty_plane_cells() if c["sep"] == 10.0]),
            VIS.build_density_chart(R.density_cells()), VIS.build_iterations_chart(R.irls_cell("Normal"))]
    for fig in figs:
        assert len(fig.data) >= 1 and _fixed(fig)


def test_heatmap_values_and_labels():
    fig = VIS.build_regret_heatmap([[0, 1.5, 20.0], [2.0, 0, 3.0], [4.0, 5.0, 0]])
    assert fig.data[0].z[0][2] == 20.0 and list(fig.data[0].y) == ["p = 1", "p = 2", "p = ∞"]
    texts = {(a.y, a.x): a.text for a in fig.layout.annotations}
    assert len(texts) == 9 and texts[("p = 1", "2 (Quadratsumme)")] == "1,5 %" and texts[("p = 1", "∞ (längster Weg)")] == "20,0 %" and texts[("p = ∞", "1 (Summe)")] == "4,0 %"
    colors = {(a.y, a.x): a.font.color for a in fig.layout.annotations}
    assert colors[("p = 1", "∞ (längster Weg)")] == "white" and colors[("p = 1", "2 (Quadratsumme)")] == "black"


def test_empty_chart_has_three_series_in_percent():
    fig = VIS.build_empty_chart([c for c in R.empty_plane_cells() if c["sep"] == 10.0])
    assert [t.name for t in fig.data] == ["p = 1", "p = 2", "p = ∞"] and list(fig.data[1].x) == ["50 %", "60 %", "70 %", "80 %"]
    assert fig.data[1].y[1] == pytest.approx(100 * R.empty_plane_value(0.6, 10.0, 2.0))


def test_compromise_chart_marks_the_best_p_with_a_star():
    cell = R.compromise_cell("Außenposten", 30)
    fig = VIS.build_compromise_chart(cell)
    star = fig.data[1]
    assert star.marker.symbol == "star" and star.y[0] == min(cell["mean"]) and star.x[0] == VIS._p_text(cell["best_p"])


def test_p_text():
    assert VIS._p_text(math.inf) == "∞" and VIS._p_text("inf") == "∞" and VIS._p_text(1.25) == "1,25" and VIS._p_text(8.0) == "8" and X.jp(math.inf) == "inf"
