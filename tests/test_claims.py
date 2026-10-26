"""Jede Zahl der README wird hier aus der vorgerechneten Messreihe (data/lpm_results.json) gebildet und in der README gesucht (DEMO-PLAYBOOK §4).

Die Datei selbst ist durch tests/test_results.py gegen die Quelldateien abgesichert; dort werden Zellen auch live nachgerechnet."""

import math
from pathlib import Path

import pytest

import lpm_constants as C
import lpm_results as R

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
INF = math.inf


def de(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return de(100 * x, nd)


def has(text):
    assert text in README, text


def row(*cells):
    return "| " + " | ".join(cells) + " |"


def test_hypothesis_table_h1_to_h4():
    r1, r2, r3 = (R.stair_row("Normal", 1.1, s)["largest_step"][0] for s in (0.1, 0.02, 0.004))
    has(f"{de(100 * r1, 2)} % / {de(100 * r2, 2)} % / {de(100 * r3, 2)} % der Spannweite bei Schrittweite 0,1 / 0,02 / 0,004")
    has(f"nur {pct(R.axis_cell('Normal', 51)['strict'][0], 0)} % (Normal), {pct(R.axis_cell('Zwei Häufungen', 51)['strict'][0], 0)} % (zwei Häufungen), {pct(R.axis_cell('Lognormal', 51)['strict'][0], 0)} % (lognormal)")
    has(f"wächst nur in {pct(R.plane_cell('Normal', 30)['sum_strict'][0], 0)} % (Normal) der Instanzen strikt mit p")
    m = R.outpost_row(6.0)["mean"]
    has(f"{de(m[1][0])} % / {de(m[1][2])} % / {de(m[0][2])} % / {de(m[2][0])} % (n = 31, {R.outpost_row(6.0)['runs_per_seed']} Instanzen je Seed)")
    has(f"das {de(1 + m[2][0] / 100, 3)}-fache")
    has(f"{de(1 + m[2][0] / 100, 3)}-mal die Summe")
    has(f"({de(m[2][0])} % mehr)")
    n30 = R.regret_cell("Außenposten", 30)["mean"]
    has(f"{de(n30[2][1])} % gegen {de(n30[1][2])} % (n = 30)")
    has(f"{de(n30[2][1])} % Mehrkosten, die Wahl nach p = 2, bewertet nach dem längsten Weg, {de(n30[1][2])} %")


def test_hypothesis_table_h5_to_h11():
    cell = R.compromise_cell("Normal", 30)
    grid = list(cell["grid"])
    has(f"{de(cell['mean'][grid.index(6.0)])} % gegen {de(cell['mean'][grid.index(8.0)])} % bei p = 8")
    has(f"{de(cell['mean'][grid.index(6.0)])} % und {de(cell['mean'][grid.index(8.0)])} %")
    assert [R.compromise_cell(lay, 30)["best_p"] for lay in C.LAYOUTS] == [6.0, 16.0, 3.0, 3.0]
    d8 = [R.approach_row(lay, 8.0)["dist_radius"] for lay in C.LAYOUTS]
    d32 = [R.approach_row(lay, 32.0)["dist_radius"] for lay in C.LAYOUTS]
    has(f"bei p = 8: {pct(min(d8))} bis {pct(max(d8))} %, bei p = 32: {pct(min(d32))} bis {pct(max(d32))} %")
    has(f"Mittel in {pct(R.empty_plane_value(0.6, 10.0, 2.0), 0)} % der Instanzen, Median in {pct(R.empty_plane_value(0.6, 10.0, 1.0), 0)} %")
    has(", ".join([f"{pct(R.density_value(3.0, 1.0), 0)} % (p = 1)", f"{pct(R.density_value(3.0, 0.7), 0)} % (0,7)", f"{pct(R.density_value(3.0, 0.5), 0)} % (0,5)", f"{pct(R.density_value(3.0, 0.4), 0)} % (0,4)"]))
    plane_half = [R.plane_data_value(lay, n, 0.5) for lay, n in (("Normal", 5), ("Normal", 10), ("Normal", 30), ("Zwei Häufungen", 30))]
    has(f"in {pct(min(plane_half), 0)} bis {pct(max(plane_half), 0)} % bei p = 0,5")
    cell7 = {lay: R.irls_cell(lay) for lay in ("Normal", "Zwei Häufungen")}
    maxit = cell7["Normal"]["maxit"]
    assert all(R.irls_row(lay, p)["plain"]["converged"] == 0.0 and R.irls_row(lay, p)["damped"]["converged"] == 1.0 for lay in cell7 for p in (4.0, 6.0, 8.0))
    has(f"bei p = 4 / 6 / 8 in 0 % der Läufe (Grenze {maxit:,} Schritte)".replace(",", "."))
    a = {p: R.start_row(p) for p in (0.7, 0.5, 0.2)}
    has(f"enden {pct(a[0.7]['B'][0], 0)} % (p = 0,7), {pct(a[0.5]['B'][0], 0)} % (0,5), {pct(a[0.2]['B'][0], 0)} % (0,2); Ebene: {pct(R.plane_start_row(0.7)['B'][0], 0)} % / {pct(R.plane_start_row(0.5)['B'][0], 0)} % / {pct(R.plane_start_row(0.2)['B'][0], 0)} %")
    o = {a_: R.outpost_row(a_)["mean"][0][2] for a_ in (2.0, 3.0, 6.0)}
    has(f"{de(o[2.0])} % (a = 2), {de(o[3.0])} % (a = 3), {de(o[6.0])} % (a = 6)")


def test_corrections_section():
    has(f"bis {de(max(c['long_back_max_pct'] for c in R.plane_cells()))} % des längsten Wegs (Quadrat, n = 30)")
    assert max(R.plane_cells(), key=lambda c: c["long_back_max_pct"])["layout"] == "Quadrat" and max(R.plane_cells(), key=lambda c: c["long_back_max_pct"])["n"] == 30
    has(f"bis {de(max(c['sum_back_max_pct'] for c in R.plane_cells() if c['layout'] == 'Zwei Häufungen'))} % der Summe (zwei Häufungen, n = 100)")
    assert R.plane_cell("Zwei Häufungen", 100)["sum_back_max_pct"] == max(c["sum_back_max_pct"] for c in R.plane_cells())
    z3 = R.irls_row("Zwei Häufungen", 3.0)
    has(f"steigt der Zielwert bei p = 3 in {pct(z3['plain']['rises'], 0)} % der Läufe")
    top = max(R.irls_row(lay, p)["damped"]["max_its"] for lay in ("Normal", "Zwei Häufungen") for p in (4.0, 6.0, 8.0))
    has(f"in höchstens {top:.0f} Schritten")
    has(f"mit vier Ziehungen sind es {pct(R.start_row(0.7)['B'][0], 0)} / {pct(R.start_row(0.5)['B'][0], 0)} / {pct(R.start_row(0.2)['B'][0], 0)} %")
    has(f"{pct(R.start_row(0.5)['A'][0], 0)} % (p = 0,5) und {pct(R.start_row(0.2)['A'][0], 0)} % (p = 0,2)")
    has(f"({de(100 * R.stair_row('Normal', 0.1, 0.004)['largest_step'][0], 2)} % der Spannweite bei Schrittweite 0,004 statt {de(100 * R.stair_row('Normal', 0.1, 0.1)['largest_step'][0], 2)} % bei 0,1)")


def test_e1_tables():
    ex = R.example_row
    has(f"p = 2 gibt {de(ex(2.0)['lo'], 3)}, p = 8 gibt {de(ex(8.0)['lo'], 3)}, p = 32 gibt {de(ex(32.0)['lo'], 3)}, p = ∞ gibt {de(ex(INF)['lo'], 3)}")
    assert (ex(1.0)["lo"], ex(1.0)["hi"]) == (5.0, 6.0)
    for layout in ("Normal", "Lognormal", "Außenposten", "Zwei Häufungen"):
        has(row(layout, *[pct(R.axis_cell(layout, n)["strict"][0], 0) for n in (9, 51, 201)]))
    has(f"sind es bis p = 8 noch {pct(R.axis_cell('Zwei Häufungen', 51)['strict_to_8'][0], 0)} %")
    has(f"im Mittel {de(100 * R.axis_cell('Normal', 51)['back_mean'], 2)} % der Spannweite, in der ungünstigsten Instanz {de(100 * R.axis_cell('Normal', 51)['back_max'], 1)} %")
    s = [pct(R.plane_cell(lay, 30)["sum_strict"][0], 0) for lay in C.LAYOUTS]
    lg = [pct(R.plane_cell(lay, 30)["long_strict"][0], 0) for lay in C.LAYOUTS]
    has(f"{s[0]} % (Normal, n = 30), {s[1]} % (Quadrat), {s[2]} % (Außenposten) und {s[3]} % (zwei Häufungen)")
    has(f"nur in {lg[0]} % / {lg[1]} % / {lg[2]} % / {lg[3]} %")
    has(f"Größter Rückschritt aller Zellen: {de(max(max(c['sum_back_max_pct'], c['long_back_max_pct']) for c in R.plane_cells()))} % des Wertes")
    for lay in C.LAYOUTS:
        label = {"Normal": "Normal (%)", "Quadrat": "Quadrat (%)", "Außenposten": "Außenposten (%)", "Zwei Häufungen": "Zwei Häufungen (%)"}[lay]
        has(row(label, *[pct(R.approach_row(lay, p)["dist_radius"]) for p in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)]))
    out = [R.approach_row("Außenposten", p) for p in (4.0, 8.0, 16.0, 32.0)]
    nor = [R.approach_row("Normal", p) for p in (4.0, 8.0, 16.0, 32.0)]
    has(f"um {de(out[0]['overshoot_pct'])} % (p = 4), {de(out[1]['overshoot_pct'])} % (p = 8), {de(out[2]['overshoot_pct'])} % (p = 16) und {de(out[3]['overshoot_pct'])} % (p = 32)")
    has(" / ".join(f"{de(r['sum_excess_pct'])} %" for r in out) + " höher")
    has("(Normal: " + " / ".join(f"{de(r['sum_excess_pct'])} %" for r in nor) + ")")


def test_e2_table_and_n_series():
    for layout, label in (("Normal", "Normal"), ("Quadrat", "Quadrat"), ("Außenposten", "Außenposten (a = 6)"), ("Zwei Häufungen", "Zwei Häufungen (70/30, Abstand 6)")):
        m = R.regret_cell(layout, 30)["mean"]
        has(row(label, de(m[0][1]), de(m[0][2]), de(m[1][0]), de(m[1][2]), de(m[2][0]), de(m[2][1])))
    has(f"Streuung zwischen den Seeds höchstens {de(max(max(max(r) for r in R.regret_cell(l, 30)['std']) for l in C.LAYOUTS))} Prozentpunkte")
    ns = (10, 30, 100, 300)
    has("Normal 1→∞ " + " / ".join(de(R.regret_cell("Normal", n)["mean"][0][2]) for n in ns) + " %")
    has("Quadrat 1→∞ " + " / ".join(de(R.regret_cell("Quadrat", n)["mean"][0][2]) for n in ns) + " %")
    has("(" + " / ".join(de(R.regret_cell("Außenposten", n)["mean"][2][1]) for n in ns) + " %)")
    has("(" + " / ".join(de(R.regret_cell("Zwei Häufungen", n)["mean"][1][0]) for n in ns) + " %)")
    assert R.regret_cell("Normal", 30)["runs_per_seed"] == 200


def test_e3_table():
    rows = R.outpost_rows()
    has(row("1→∞", *[de(r["mean"][0][2]) for r in rows]))
    has(row("∞→2", *[de(r["mean"][2][1]) for r in rows]))
    assert [r["a"] for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
    m = R.outpost_row(6.0)["mean"]
    has(f"Summe im Schwerpunkt {de(m[1][0])} % über dem geometrischen Median, längster Weg im Schwerpunkt {de(m[1][2])} % und im geometrischen Median {de(m[0][2])} % über dem Radius")
    assert R.outpost_row(6.0)["n"] == 31 and R.outpost_row(6.0)["runs_per_seed"] == 150


def test_e4_table():
    for layout in C.LAYOUTS:
        c = R.compromise_cell(layout, 30)
        grid = list(c["grid"])
        best = "∞" if c["best_p"] == "inf" else f"{c['best_p']:g}"
        has(row(layout, best, de(c["best_value"]), de(c["free_choice"]), de(c["mean"][0]), de(c["mean"][grid.index(2.0)]), de(c["mean"][-1])))
    fmt = lambda v: "∞" if v == "inf" else f"{v:g}"
    for layout in C.LAYOUTS:
        text = " / ".join(fmt(R.compromise_cell(layout, n)["best_p"]) for n in (10, 30, 100))
        has(f"{layout} {text}" if layout != "Zwei Häufungen" else f"zwei Häufungen {text}")
    assert R.compromise_cell("Normal", 30)["runs_per_seed"] == 150


def test_e5_tables():
    for sep_row, p in (("p = 1", 1.0), ("p = 2", 2.0), ("p = ∞", INF)):
        has(row(f"Abstand 10 km, {sep_row} (%)", *[pct(R.empty_plane_value(s, 10.0, p), 0) for s in (0.5, 0.6, 0.7, 0.8)]))
    far = max(r["mean"] for c in R.empty_plane_cells() if c["sep"] == 6.0 for r in c["rows"])
    has(f"Bei Abstand 6 km liegt keine der drei Mitten in mehr als {pct(far, 0)} % der Instanzen in der leeren Zone")
    ax = R.empty_axis_value
    has(f"Bei 55 % in der großen Häufung liegt c(p) für p = 1,5 / 2 / 3 / 4 in {pct(ax(0.55, 1.5), 0)} / {pct(ax(0.55, 2.0), 0)} / {pct(ax(0.55, 3.0), 0)} / {pct(ax(0.55, 4.0), 0)} %")
    has(f"bei 60 % für p = 2 / 4 in {pct(ax(0.6, 2.0), 0)} / {pct(ax(0.6, 4.0), 0)} %, bei 80 % für p = 4 in {pct(ax(0.8, 4.0), 0)} %")
    assert all(ax(s, 0.5) == 0.0 for s in (0.5, 0.55, 0.6, 0.7, 0.8))


def test_e6_tables():
    ps = (2.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1)
    for s in (2.0, 3.0, 4.0):
        has(row(f"s = {s:g} (%)", *[pct(R.density_value(s, p), 0) for p in ps]))
    assert all(R.density_value(s, p) == 0.0 for s in (0.5, 1.0) for p in ps)
    has(f"Bei s = 3 liegt die Mitte für p = 1 / 0,7 / 0,5 / 0,4 / 0,3 in " + " / ".join(pct(R.plane_density_value(3.0, p), 0) for p in (1.0, 0.7, 0.5, 0.4, 0.3)) + " % der Instanzen")
    assert all(R.plane_density_value(1.0, p) == 0.0 for p in (2.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2))
    for layout, label, lo in (("Normal", "Normal, 0,1 bis 0,9", 0.1), ("Normal", "Normal, 1,1 bis 3", 1.1), ("Zwei Häufungen", "Zwei Häufungen, 0,1 bis 0,9", 0.1), ("Zwei Häufungen", "Zwei Häufungen, 1,1 bis 3", 1.1)):
        has(row(label, *[de(100 * R.stair_row(layout, lo, s)["largest_step"][0], 2) for s in (0.1, 0.02, 0.004)]))
    sp = (2.0, 0.95, 0.9, 0.7, 0.5, 0.2)
    has(row("B (%)", *[pct(R.start_row(p)["B"][0], 0) for p in sp]))
    has(row("A (%)", *[pct(R.start_row(p)["A"][0], 0) for p in sp]))
    has(row("Enden auf einem Kunden (%)", *[pct(R.start_row(p)["D"][0], 0) for p in sp]))
    has(row("verschiedene Enden", *[de(R.start_row(p)["distinct"][0], 0) for p in sp]))
    has(f"Streuung {pct(R.start_row(0.7)['B'][1], 0)} Prozentpunkte")
    pd = lambda lay, n: " / ".join(pct(R.plane_data_value(lay, n, p), 0) for p in (0.9, 0.7, 0.5, 0.3, 0.1))
    has(f"{pd('Normal', 5)} % (Normal, n = 5), {pd('Normal', 10)} % (n = 10) und {pd('Normal', 30)} % (n = 30)")
    assert R.start_row(0.7)["starts"] == 200 and R.start_row(0.7)["instances_per_seed"] == 6 and R.plane_start_row(0.7)["instances_per_seed"] == 3
    assert R.plane_start_row(0.7)["starts"] == 100 and R.plane_data_cells()[0]["runs_per_seed"] == 50 and R.density_cells()[0]["runs_per_seed"] == 100


def test_e7_table():
    n, z = R.irls_cell("Normal"), R.irls_cell("Zwei Häufungen")
    ps = (1.5, 2.5, 3.0, 4.0, 6.0, 8.0)
    has(row("ohne Dämpfung: konvergiert (%)", *[pct(R.irls_row("Normal", p)["plain"]["converged"], 0) for p in ps]))
    has(row("ohne Dämpfung: Zielwert steigt (%, zwei Häufungen)", *[pct(R.irls_row("Zwei Häufungen", p)["plain"]["rises"], 0) for p in ps]))
    has(row("ohne Dämpfung: Schritte (Median, Normal)", *[(f"{R.irls_row('Normal', p)['plain']['median_its']:.0f}" if R.irls_row("Normal", p)["plain"]["median_its"] is not None else "–") for p in ps]))
    has(row("mit Dämpfung: Schritte (Median, Normal)", *[f"{R.irls_row('Normal', p)['damped']['median_its']:.0f}" for p in ps]))
    has(row("mit Dämpfung: Schritte (Median, zwei Häufungen)", *[f"{R.irls_row('Zwei Häufungen', p)['damped']['median_its']:.0f}" for p in ps]))
    has(row("Newton: Schritte (Median, Normal)", *[f"{R.irls_row('Normal', p)['newton_median_its']:.0f}" for p in ps]))
    assert all(R.irls_row(lay, p)["damped"]["rises"] == 0.0 for lay in ("Normal", "Zwei Häufungen") for p in ps)
    has(f"im Median {R.irls_row('Zwei Häufungen', 3.0)['plain']['median_its']:.0f} Schritte")
    assert n["runs_per_seed"] * len(R.meta()["seeds"]) == 160 and z["maxit"] == 5000
    has("160 Instanzen je Layout und p")
    assert C.IRLS_MAXIT_LIVE == 3000


def test_app_text_and_readme_agree_on_the_assumptions_table():
    """Die Annahmen stehen in App und README gleichlautend in den Kernpunkten (Zwei Fassungen desselben Textes werden verglichen, nicht von Hand synchron gehalten)."""
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    for phrase in ("Luftlinie in der Ebene", "Kompromiss-p nur für ein Raster", "Konvergenz nur gemessen"):
        assert phrase in app
    assert "Luftlinie in der Ebene" in README and "Konvergenz nur gemessen" in README and "Kompromiss-p nur für ein Raster" in README
