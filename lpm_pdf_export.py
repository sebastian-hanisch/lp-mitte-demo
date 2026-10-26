"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Mitten im Vergleich, Kosten, Verdikte und eine einfache Karte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

import numpy as np
from fpdf import FPDF

import lpm_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "√": "Wurzel", "‖": "|", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich", "→": "->", "ε": "eps"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def centre_rows(a: V.Analysis) -> list[tuple[str, np.ndarray]]:
    """(Name, Position) der Mitten, die die Karte zeigt."""
    return [("Schwerpunkt (p = 2)", a.centroid), ("Geometrischer Median (p = 1)", a.median), ("1-Center (p = unendlich)", a.one_center),
            (f"Gewählte Mitte (p = {V.fmt_p(a.settings.p)})", a.reference)]


def build_pdf(a: V.Analysis, verdicts: list[tuple[str, str]]) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Die Mitte hängt vom Strafmaß ab"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Layout {s.layout}, n = {s.n}, Exponent p = {V.fmt_p(s.p)}, Löser {s.solver}, Start {s.start}, Zufalls-Seed {s.seed}. "
                                f"Außenposten a = {s.outpost:g}, Anteil der großen Häufung {s.share} %, Abstand {s.sep:g} km, Streuung {s.spread:g}."))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Mitten und ihre Kosten", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    import lpm_center as K
    for name, c in centre_rows(a):
        costs = [K.lp_cost(a.points, c, q) for q in (1.0, 2.0, float("inf"))]
        pdf.cell(0, 5.5, _latin(f"{name}: ({V.de(c[0], 3)}; {V.de(c[1], 3)}), Summe {V.de(costs[0], 2)}, Quadratsumme {V.de(costs[1], 2)}, längster Weg {V.de(costs[2], 2)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text))
        pdf.ln(1)
    x0, y0, size = 30.0, pdf.get_y() + 4, 120.0
    if y0 + size > 280:
        pdf.add_page()
        y0 = 20.0
    pts = a.points
    lo = np.minimum(pts.min(axis=0), a.one_center - a.one_radius)
    hi = np.maximum(pts.max(axis=0), a.one_center + a.one_radius)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-9))

    def to_page(q):
        return x0 + (q[0] - lo[0]) / span * size, y0 + size - (q[1] - lo[1]) / span * size

    pdf.set_draw_color(180, 180, 180)
    pdf.rect(x0 - 3, y0 - 3, size + 6, size + 6)
    pdf.set_fill_color(110, 110, 110)
    for q in pts:
        px, py = to_page(q)
        pdf.ellipse(px - 0.9, py - 0.9, 1.8, 1.8, style="F")
    cx, cy = to_page(a.one_center)
    r = a.one_radius / span * size
    pdf.set_draw_color(148, 103, 189)
    pdf.ellipse(cx - r, cy - r, 2 * r, 2 * r)
    for name, c in centre_rows(a):
        px, py = to_page(c)
        pdf.set_draw_color(214, 39, 40) if name.startswith("Gewählte") else pdf.set_draw_color(47, 107, 101)
        pdf.line(px - 2, py - 2, px + 2, py + 2)
        pdf.line(px - 2, py + 2, px + 2, py - 2)
    pdf.set_y(y0 + size + 6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Punkte: Kunden; Kreis: kleinster umschließender Kreis; Kreuze: rot = gewählte Mitte, grün = Schwerpunkt, geometrischer Median und 1-Center. Luftlinie, Einheit km."))
    return bytes(pdf.output())
