"""PDF-Export: gültige Datei, alle Zeichen aus dem Latin-1-Satz, Kernschriften stürzen nicht ab."""

import math

import lpm_constants as C
import lpm_evaluation as V
from lpm_pdf_export import _latin, build_pdf, centre_rows


def test_pdf_is_built_for_every_regime():
    for p in (0.4, 1.0, 2.0, 6.0, math.inf):
        a = V.analyse(V.Settings(p=p, layout="Außenposten", solver=C.SOLVERS[1]))
        data = build_pdf(a, [V.verdict_regret(a), V.verdict_solver(a)])
        assert data[:5] == b"%PDF-" and len(data) > 1500, p


def test_latin_replaces_special_characters_and_markdown():
    assert _latin("p = ∞ → „x“ ≥ 2 · 3 – 4 **fett** `c`") == 'p = unendlich -> "x" >= 2 * 3 - 4 fett c'
    assert _latin("Größe ä ö ü ß") == "Größe ä ö ü ß"


def test_centre_rows_names_the_four_marks():
    a = V.analyse(V.Settings(p=3.0))
    rows = centre_rows(a)
    assert [name for name, _ in rows] == ["Schwerpunkt (p = 2)", "Geometrischer Median (p = 1)", "1-Center (p = unendlich)", "Gewählte Mitte (p = 3)"]
    assert (rows[3][1] == a.reference).all()
