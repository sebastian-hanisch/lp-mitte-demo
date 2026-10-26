"""Plotly-Figuren: Karte mit Mitten, Pfad und 1-Center-Kreis, Achsenpfad, Diagramme der Experimente. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängt."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import lpm_constants as C

FONT = dict(size=13)
INF_LABEL = "∞"


def _p_text(p) -> str:
    return INF_LABEL if p == math.inf or p == "inf" else f"{float(p):g}".replace(".", ",")


def _base(fig: go.Figure, height: int, xtitle: str, ytitle: str) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=55, r=15, t=30, b=45), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), legend=dict(orientation="h", y=-0.22), font=FONT)
    return fig


def _circle(center, radius, count=90):
    t = np.linspace(0, 2 * np.pi, count)
    return center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)


def build_map(a) -> go.Figure:
    """Karte der Kunden, der drei Marken (Schwerpunkt p = 2, geometrischer Median p = 1, 1-Center p = ∞ samt Kreis), des Pfades c(p) über p ≥ 1 und der gewählten Mitte."""
    p = a.points
    fig = go.Figure()
    labels = [str(i + 1) for i in range(len(p))] if len(p) <= 30 else None
    fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="markers+text" if labels else "markers", text=labels, textposition="middle center", textfont=dict(size=9, color="white"),
                             marker=dict(size=14 if labels else 7, color=C.COLOR_POINT, line=dict(width=1, color="white")), name="Kunden",
                             hovertemplate="(%{x:.2f}; %{y:.2f})<extra>Kunde</extra>"))
    cx, cy = _circle(a.one_center, a.one_radius)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=C.COLOR_ONE_CENTER, dash="dash", width=1.5), name="Kleinster umschließender Kreis", hoverinfo="skip"))
    path = a.path
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines+markers+text", line=dict(color=C.COLOR_PATH, width=1.5), marker=dict(size=5),
                             text=[_p_text(q) if q in (1.0, 2.0, 4.0, 8.0, 16.0, 64.0, math.inf) else "" for q in a.path_p], textposition="top center", textfont=dict(size=10, color=C.COLOR_PATH),
                             name="Pfad der Mitte für p = 1 bis ∞",
                             hovertemplate="p = %{customdata}<extra></extra>", customdata=[_p_text(q) for q in a.path_p]))

    def mark(name, c, symbol, color, size, hollow=False):
        marker = dict(symbol=symbol, size=size, color="rgba(0,0,0,0)" if hollow else color, line=dict(width=2.5 if hollow else 1.5, color=color))
        fig.add_trace(go.Scatter(x=[c[0]], y=[c[1]], mode="markers", marker=marker, name=name))

    mark("Schwerpunkt (p = 2)", a.centroid, "diamond", C.COLOR_MEAN, 13)
    mark("Geometrischer Median (p = 1)", a.median, "square", C.COLOR_MEDIAN, 12)
    mark("1-Center (p = ∞)", a.one_center, "x", C.COLOR_ONE_CENTER, 13)
    mark(f"Gewählte Mitte (p = {_p_text(a.settings.p)})", a.reference, "star", C.COLOR_CENTER, 18)
    if a.solver_gap > 1e-6:
        mark("Ergebnis des Lösers", a.center, "star", C.COLOR_CENTER, 18, hollow=True)
    lo = np.minimum(p.min(axis=0), a.one_center - a.one_radius)
    hi = np.maximum(p.max(axis=0), a.one_center + a.one_radius)
    pad = 0.08 * float(max(np.ptp([lo, hi], axis=0).max(), 1e-9))
    fig.add_trace(go.Scatter(x=[lo[0] - pad, hi[0] + pad], y=[lo[1] - pad, hi[1] + pad], mode="markers", marker=dict(opacity=0), showlegend=False, hoverinfo="skip"))
    _base(fig, 560, "x (km)", "y (km)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def build_axis_path(x: np.ndarray, ps, intervals) -> go.Figure:
    """Achsenpfad: waagerecht die Lage der Mitte c(p), senkrecht der Exponent p (logarithmisch); unten die Werte als Striche. `intervals`: Liste (lo, hi) je p."""
    ps, intervals = list(ps), list(intervals)
    finite = [i for i, q in enumerate(ps) if q != math.inf]
    mids = [(intervals[i][0] + intervals[i][1]) / 2 for i in finite]
    ps_finite = [ps[i] for i in finite]
    fig = go.Figure()
    floor = min(ps_finite) * 0.8
    fig.add_trace(go.Scatter(x=list(x), y=[floor] * len(x), mode="markers", marker=dict(symbol="line-ns", size=14, color=C.COLOR_POINT, line=dict(width=1.5, color=C.COLOR_POINT)), name="Werte",
                             hovertemplate="%{x:.2f}<extra>Wert</extra>"))
    fig.add_trace(go.Scatter(x=mids, y=ps_finite, mode="lines+markers", line=dict(color=C.COLOR_PATH), name="Mitte c(p)", hovertemplate="p = %{y:g}: c = %{x:.3f}<extra></extra>"))
    if 1.0 in ps:
        lo_i, hi_i = intervals[ps.index(1.0)]
        if hi_i > lo_i:
            fig.add_trace(go.Scatter(x=[lo_i, hi_i], y=[1.0, 1.0], mode="lines", line=dict(color=C.COLOR_CENTER, width=6), name="Median-Intervall (p = 1)", hoverinfo="skip"))
    if math.inf in ps:
        x_inf = intervals[ps.index(math.inf)][0]
        fig.add_shape(type="line", x0=x_inf, x1=x_inf, y0=0, y1=1, yref="paper", line=dict(color=C.COLOR_ONE_CENTER, dash="dash"))
        fig.add_annotation(x=x_inf, y=1, yref="paper", text="p = ∞ (Mitte der Spannweite)", showarrow=False, xanchor="right", yanchor="top", xshift=-4)
    _base(fig, 420, "Lage auf der Achse", "Exponent p")
    ticks = [q for q in (0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64) if q >= floor]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{q:g}".replace(".", ",") for q in ticks])
    return fig


def build_example_path(rows: list[dict]) -> go.Figure:
    """Zahlenbeispiel: c(p) für die Werte 1 bis 9 und 40 gegen p (p = ∞ am rechten Rand als gestrichelte Linie)."""
    finite = [r for r in rows if r["p"] != "inf"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["p"] for r in finite], y=[(r["lo"] + r["hi"]) / 2 for r in finite], mode="lines+markers", line=dict(color=C.COLOR_PATH), name="Mitte c(p)"))
    p1 = finite[0]
    fig.add_trace(go.Scatter(x=[1, 1], y=[p1["lo"], p1["hi"]], mode="lines", line=dict(color=C.COLOR_CENTER, width=6), name="Median-Intervall (p = 1)"))
    last = rows[-1]
    fig.add_hline(y=last["lo"], line_dash="dash", line_color=C.COLOR_ONE_CENTER, annotation_text="p = ∞: Mitte der Spannweite", annotation_position="bottom right")
    _base(fig, 380, "Exponent p", "Lage der Mitte")
    fig.update_xaxes(type="log")
    return fig


def build_approach_chart(cells: list[dict]) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        fig.add_trace(go.Scatter(x=[r["p"] for r in cell["rows"]], y=[r["dist_radius"] for r in cell["rows"]], mode="lines+markers", name=cell["layout"]))
    _base(fig, 380, "Exponent p", "Abstand von c(p) zum 1-Center in Radien")
    fig.update_xaxes(type="log", tickvals=[4, 8, 16, 32, 64, 128], ticktext=["4", "8", "16", "32", "64", "128"])
    return fig


def build_regret_heatmap(mean, title: str = "") -> go.Figure:
    """Regret-Matrix (Zeile: gewählt nach p, Spalte: bewertet nach q), Zahlen in Prozent."""
    labels = ["1 (Summe)", "2 (Quadratsumme)", "∞ (längster Weg)"]
    z = np.asarray(mean, dtype=float)
    rows = ["p = 1", "p = 2", "p = ∞"]
    top = max(float(z.max()), 1.0)
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=rows, colorscale="Oranges", zmin=0, zmax=top, showscale=False, hovertemplate="gewählt %{y}, bewertet nach q = %{x}: %{z:.1f} %<extra></extra>"))
    for r, row in enumerate(z):                       # Zahlen als Annotationen (nicht als Text der Heatmap: kein Hintergrund und Layoutfehler vor dem ersten Größenwechsel)
        for c, v in enumerate(row):
            fig.add_annotation(x=labels[c], y=rows[r], text=f"{v:.1f} %".replace(".", ","), showarrow=False, font=dict(color="white" if v > 0.6 * top else "black", size=14))
    _base(fig, 280, "bewertet nach q", "gewählt nach p")
    fig.update_yaxes(autorange="reversed")
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return fig


def build_regret_by_n(cells: list[dict], column: int, row: int, title: str) -> go.Figure:
    """Ein Matrixeintrag über n, eine Linie je Layout."""
    fig = go.Figure()
    for layout in dict.fromkeys(c["layout"] for c in cells):
        sub = [c for c in cells if c["layout"] == layout]
        fig.add_trace(go.Scatter(x=[c["n"] for c in sub], y=[c["mean"][row][column] for c in sub], mode="lines+markers", name=layout,
                                 error_y=dict(type="data", array=[c["std"][row][column] for c in sub], visible=True, thickness=1)))
    _base(fig, 340, "Kunden n", title)
    fig.update_xaxes(type="log", tickvals=[10, 30, 100, 300], ticktext=["10", "30", "100", "300"])
    return fig


def build_outpost_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    series = ((0, 2, "p = 1 → längster Weg"), (2, 1, "p = ∞ → Quadratsumme"), (1, 0, "p = 2 → Summe"))
    for r_, c_, name in series:
        fig.add_trace(go.Scatter(x=[r["a"] for r in rows], y=[r["mean"][r_][c_] for r in rows], mode="lines+markers", name=name))
    _base(fig, 380, "Abstand a des Außenpostens (bei (a; a))", "Regret (%)")
    return fig


def build_compromise_chart(cell: dict, others: list[dict] | None = None) -> go.Figure:
    grid = [_p_text(p) for p in cell["grid"]]
    fig = go.Figure()
    for c in [cell] + (others or []):
        fig.add_trace(go.Scatter(x=grid, y=c["mean"], mode="lines+markers", name=c["layout"] + (f", n = {c['n']}" if others else ""),
                                 error_y=dict(type="data", array=c["std"], visible=True, thickness=1)))
        best = list(c["grid"]).index(c["best_p"])
        fig.add_trace(go.Scatter(x=[grid[best]], y=[c["mean"][best]], mode="markers", marker=dict(symbol="star", size=14, color="black"), showlegend=False,
                                 hovertemplate=f"bestes festes p = {_p_text(c['best_p'])}<extra></extra>"))
    _base(fig, 380, "Exponent p", "größter Regret über die drei Kriterien (%)")
    return fig


def build_empty_chart(rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for key, name in ((1.0, "p = 1"), (2.0, "p = 2"), ("inf", "p = ∞")):
        fig.add_trace(go.Bar(x=[f"{round(100 * r['share'])} %" for r in rows], y=[100 * next(x["mean"] for x in r["rows"] if x["p"] == key) for r in rows], name=name))
    _base(fig, 340, "Anteil der großen Häufung", "Mitte in der leeren Zone (%)")
    fig.update_layout(barmode="group")
    return fig


def build_density_chart(cells: list[dict]) -> go.Figure:
    fig = go.Figure()
    for cell in cells:
        fig.add_trace(go.Scatter(x=[_p_text(r["p"]) for r in cell["rows"]], y=[100 * r["mean"] for r in cell["rows"]], mode="lines+markers", name=f"Streuung der großen Häufung {cell['s_big']:g}".replace(".", ",")))
    _base(fig, 380, "Exponent p", "Mitte in der dichten, kleinen Häufung (%)")
    return fig


def build_iterations_chart(cell: dict) -> go.Figure:
    fig = go.Figure()
    rows = cell["rows"]
    ps = [f"{r['p']:g}".replace(".", ",") for r in rows]
    fig.add_trace(go.Scatter(x=ps, y=[r["newton_median_its"] for r in rows], mode="lines+markers", name="Newton (Median)", line=dict(color=C.COLOR_MEDIAN)))
    fig.add_trace(go.Scatter(x=ps, y=[r["damped"]["median_its"] for r in rows], mode="lines+markers", name="Gewichtsiteration mit Dämpfung (Median)", line=dict(color=C.COLOR_PATH)))
    fig.add_trace(go.Scatter(x=ps, y=[r["plain"]["median_its"] for r in rows], mode="lines+markers", name="Gewichtsiteration ohne Dämpfung (Median, nur Läufe, die konvergieren)", line=dict(color=C.COLOR_LINE2)))
    _base(fig, 380, "Exponent p", "Iterationen bis Schritt < 1e-10")
    fig.update_yaxes(type="log")
    return fig
