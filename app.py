"""Die Mitte hängt vom Strafmaß ab (Lp-Mitte) - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 3 der Reihe „Mittelwerte und Lagemaße“. Lauffähig mit: streamlit run app.py
"""

import math

import numpy as np
import streamlit as st

import lpm_center as K
import lpm_constants as C
import lpm_evaluation as V
import lpm_experiments as X
import lpm_results as R
from lpm_layout import dunkles_design, kopf_css
from lpm_pdf_export import build_pdf, centre_rows
from lpm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lpm_scenario import AXIS_LAYOUTS, Draws, axis_batch, cell_seed
from lpm_visualization import (
    build_approach_chart,
    build_axis_path,
    build_compromise_chart,
    build_density_chart,
    build_empty_chart,
    build_example_path,
    build_iterations_chart,
    build_map,
    build_outpost_chart,
    build_regret_by_n,
    build_regret_heatmap,
)

st.set_page_config(page_title="Die Mitte hängt vom Strafmaß ab – Sebastian Hanisch", layout="wide")

AXIS_LIVE_P = tuple(round(0.1 * k, 1) for k in range(1, 10)) + X.AXIS_P
EXAMPLE_VALUES = "Zahlenbeispiel (1 bis 9 und 40)"


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return V.analyse(settings)


@st.cache_data(show_spinner=False)
def _regret_live(layout, n, runs, outpost, share, sep, spread):
    return X.regret_cell(layout, n, runs, outpost=outpost, share=share, sep=sep, spread=spread)


@st.cache_data(show_spinner=False)
def _compromise_live(layout, n, runs, outpost, share, sep, spread):
    return X.compromise_cell(layout, n, runs, outpost=outpost, share=share, sep=sep, spread=spread)


@st.cache_data(show_spinner=False)
def _irls_live(settings):
    points = V.build_points(settings)
    return [{"p": p, **X.irls_instance(points, p, C.IRLS_MAXIT_LIVE)} for p in (3.0, 4.0, 6.0, 8.0)]


@st.cache_data(show_spinner=False)
def _starts_live(settings):
    points = V.build_points(settings)
    lo, hi = points.min(axis=0), points.max(axis=0)
    starts = lo + Draws(cell_seed(settings.seed, "starts")).uniform((C.STARTS_BELOW_ONE, 2)) * (hi - lo)
    ends, its = K.irls_multi(points, settings.p, starts, maxit=C.IRLS_MAXIT_LIVE)
    ref = K.best_below_one(points, settings.p, settings.seed, C.STARTS_BELOW_ONE)
    costs = (np.linalg.norm(points[None] - ends[:, None], axis=2) ** settings.p).sum(axis=1)
    return {"distinct": len(np.unique(np.round(ends, 5), axis=0)), "on_data": float(np.mean(np.abs(ends[:, None] - points[None]).sum(axis=2).min(axis=1) < 1e-6)),
            "best_share": float(np.mean(np.linalg.norm(ends - ref.center, axis=1) <= 0.2)), "worst_excess": float(100 * (costs.max() / ref.cost - 1)), "starts": len(ends)}


def _axis_values(kind, n, seed):
    if kind == EXAMPLE_VALUES:
        return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 40], dtype=float)
    return axis_batch(Draws(cell_seed(seed, "axislive", kind, n)), kind, 1, n)[0]


def _verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


def _fmt_table(rows, **formats):
    st.dataframe(rows, hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format=v) for k, v in formats.items()})


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lpm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📏 Die Mitte hängt vom Strafmaß ab")
st.markdown(
    """
**Stück 3 der Reihe „Mittelwerte und Lagemaße“.** Ein Lager soll Kunden in der Ebene beliefern. Wo es steht, hängt davon ab, **wie ein langer Weg bestraft wird**: Die **Lp-Mitte** minimiert die Summe der Wege hoch **p**.
Bei p = 2 ist das der Schwerpunkt, bei p = 1 der geometrische Median, für sehr große p der Mittelpunkt des kleinsten umschließenden Kreises, und für p < 1 zählt fast nur noch, *wer nicht dort ist*.
Die Demo zeigt, **wohin die Mitte läuft**, **was die Wahl in den anderen Kriterien kostet** (die Regret-Matrix), warum es **kein p gibt, das überall gut ist**, wie die Mitte für p < 1 zur dichtesten Stelle springt und warum die **Gewichtsiteration** ab p = 4 nicht mehr konvergiert.
Abgrenzung: Die [p-center-demo](https://github.com/sebastian-hanisch/p-center-demo) wählt *mehrere* Mittelpunkte aus diskreten Kandidaten (dort ist p die *Zahl der Mittelpunkte*), die [kmeans-demo](https://github.com/sebastian-hanisch/kmeans-demo) vergleicht Mittelwert und Medoid, die [standortplanung-demo](https://github.com/sebastian-hanisch/standortplanung-demo) arbeitet mit diskreten Kandidaten.

**Der Buchstabe p.** In dieser Demo heißt **p immer der Exponent der Strecke**; die Zahl der Mittelpunkte kommt hier nicht vor (es gibt genau eine Mitte), und das Maximum-Ende heißt „1-Center“, nicht „p-Center“. Das Bewertungskriterium trägt den Buchstaben **q**.

Weiter unten: sieben Experimente (**🔬 E1 bis E7**), **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Lp-Mitte", expanded=True):
    st.markdown(
        r"""
1. **Was gesucht ist.** Ein Punkt $c$, der $\sum_i \|x_i - c\|^p$ klein macht: die Summe aller Kunden-Wege, jeder Weg hoch $p$ (gleiche Gewichte, Luftlinie).
2. **Drei bekannte Stellen.** $p = 2$: der **Schwerpunkt** (Mittelwert). $p = 1$: der **geometrische Median** (Weber-Punkt, Stück 5 der Reihe). $p \to \infty$: die Mitte des **kleinsten umschließenden Kreises** (1-Center), der Punkt, der den *längsten* Weg klein macht.
3. **Dazwischen ein Pfad.** Für $p \ge 1$ gibt es genau eine Mitte $c(p)$, und sie läuft stetig mit $p$. Sie läuft aber **nicht immer in eine Richtung** (E1).
4. **Drei Kriterien.** Wie gut eine Mitte ist, hängt davon ab, woran man sie misst: Summe der Wege ($q = 1$), Quadratsumme ($q = 2$) oder längster Weg ($q = \infty$). Jede Wahl von $p$ ist für ein anderes $q$ optimal; die **Regret-Matrix** zeigt, wie viel Mehrkosten die Wahl in den anderen Kriterien kostet (E2).
5. **Unter 1 wird es anders.** Für $p < 1$ ist die Zielfunktion nicht mehr konvex: Jeder Kunde ist ein lokales Minimum, die Mitte ist meist ein Kunde und springt zur **dichtesten** Stelle (E6).
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🏭 Kunden")
    layout = st.selectbox("Layout", C.LAYOUTS, key="layout_select",
                          help="Wie die Kunden in der Ebene liegen: normalverteilt, gleichmäßig im Quadrat, ein Außenposten bei (a; a) oder zwei Häufungen.")
    n = st.slider("Kunden n", *bounds("n_slider"), key="n_slider", help="Anzahl der Kunden, alle mit gleichem Gewicht.")
    outpost = st.slider("Abstand a des Außenpostens", *bounds("outpost_slider"), key="outpost_slider", step=0.5, disabled=layout != "Außenposten",
                        help="Nur im Layout „Außenposten“: Ein Kunde steht bei (a; a), die übrigen n − 1 sind normalverteilt.")
    share = st.slider("Anteil der großen Häufung (%)", *bounds("share_slider"), key="share_slider", step=5, disabled=layout != "Zwei Häufungen",
                      help="Nur im Layout „Zwei Häufungen“: Anteil der Kunden in der großen Häufung um (0; 0).")
    sep = st.slider("Abstand der Häufungen (km)", *bounds("sep_slider"), key="sep_slider", step=0.5, disabled=layout != "Zwei Häufungen",
                    help="Nur im Layout „Zwei Häufungen“: Die kleine Häufung liegt um (Abstand; 0).")
    spread = st.slider("Streuung der großen Häufung", *bounds("spread_slider"), key="spread_slider", step=0.5, disabled=layout != "Zwei Häufungen",
                       help="Nur im Layout „Zwei Häufungen“: Standardabweichung der großen Häufung; die kleine hat 1. Je größer, desto dünner steht die große Häufung (E6).")
    st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.header("📏 Strafmaß")
    p = st.select_slider("Exponent p der Strecke", C.P_OPTIONS, key="p_select", format_func=V.fmt_p,
                         help="Die Mitte minimiert die Summe der Wege hoch p. p = 2: Schwerpunkt, p = 1: geometrischer Median, p = ∞: Mitte des kleinsten umschließenden Kreises, p < 1: meist ein Kunde. "
                              "(Hier ist p der Exponent; in der p-center-demo bedeutet p die Zahl der Mittelpunkte.)")
    criterion = st.selectbox("Bewertet nach Kriterium q", C.CRITERIA, key="criterion_select",
                             help="Nach welchem Kriterium die Kennzahl „Regret“ und die Hervorhebung bewertet: Summe der Wege, Quadratsumme oder längster Weg.")
    st.header("⚙️ Löser")
    solver_active = 1.0 < p < math.inf and p != 2.0
    solver = st.selectbox("Löser", C.SOLVERS, key="solver_select", disabled=not solver_active,
                          help="Nur für 1 < p < ∞ außer p = 2: Newton-Verfahren, Gewichtsiteration ohne Dämpfung oder mit Dämpfung 1/(p − 1). Für p = 1 rechnet Vardi–Zhang, für p = 2 die Formel, für p = ∞ der kleinste umschließende Kreis, für p < 1 die Gewichtsiteration.")
    start = st.selectbox("Start", C.STARTS, key="start_select", disabled=p in (2.0, math.inf),
                         help="Startpunkt der Iteration: der Schwerpunkt oder ein Zufallspunkt im Rahmen der Kunden. Für p ≥ 1 führt jeder Start zur selben Mitte, für p < 1 nicht (E6). Für p = 2 und p = ∞ gibt es keine Iteration.")

settings = V.Settings(layout, int(n), float(outpost), int(share), float(sep), float(spread), float(p), criterion, solver, start, int(st.session_state["seed_input"]))
sync_query_params({k: st.session_state[k] for k in ("layout_select", "n_slider", "outpost_slider", "share_slider", "sep_slider", "spread_slider", "p_select", "criterion_select",
                                                      "solver_select", "start_select", "seed_input")})

with st.spinner("Rechne …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

q_sel = V.criterion_q(criterion)
q_index = list(C.CRITERION_Q).index(q_sel)
st.markdown(f"## 🎯 Ergebnis: Mitte für p = {V.fmt_p(p)}")
cols = st.columns(4)
for i, (col, name) in enumerate(zip(cols[:3], ("Summe der Wege", "Quadratsumme", "längster Weg"))):
    col.metric(name + (" (gewählt)" if i == q_index else ""), V.de(a.costs[i], 2), delta=f"{V.de(a.regrets[i], 1)} % über dem Optimum", delta_color="inverse" if a.regrets[i] > 0.05 else "off")
cols[3].metric("Nächster Kunde", f"{V.de(a.nearest, 2)} km", delta="Abstand der Mitte", delta_color="off")
st.plotly_chart(build_map(a), width="stretch", key="map_chart")
st.caption("Blau: der Pfad der Mitte c(p) für p = 1 bis ∞ (Beschriftung bei p = 1, 2, 4, 8, 16, 64, ∞). Gestrichelt: der kleinste umschließende Kreis (Mitte = 1-Center). "
           "Roter Stern: die Mitte zum gewählten p; ein hohler Stern zeigt, wo der gewählte Löser stehen bleibt, wenn er davon abweicht.")
rows = []
for name, c in centre_rows(a):
    rows.append({"Mitte": name, "x": c[0], "y": c[1], **{label: K.lp_cost(a.points, c, q) for label, q in (("Summe", 1.0), ("Quadratsumme", 2.0), ("längster Weg", math.inf))}})
_fmt_table(rows, x="%.3f", y="%.3f", Summe="%.3f", Quadratsumme="%.3f", **{"längster Weg": "%.3f"})

# PDF erst auf Klick
pdf_key = repr(settings)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf"] = (pdf_key, build_pdf(a, [V.verdict_regret(a), V.verdict_solver(a)]))
if st.session_state.get("pdf", (None, None))[0] == pdf_key:
    st.download_button("⬇️ PDF herunterladen", data=st.session_state["pdf"][1], file_name="lp-mitte.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wohin läuft die Mitte, und was kostet die Wahl?")
_verdict(V.verdict_regret(a))
_verdict(V.verdict_solver(a))
if p < 1.0 and n <= C.BELOW_ONE_LIVE_MAX_N:
    if st.button(f"{C.STARTS_BELOW_ONE} Zufallsstarts auf dieser Instanz (p = {V.fmt_p(p)})", key="starts_live"):
        with st.spinner("Rechne 20 Starts …"):
            st.session_state["starts_result"] = (repr(settings), _starts_live(settings))
    res = st.session_state.get("starts_result")
    if res is not None and res[0] == repr(settings):
        r = res[1]
        st.info(f"{r['starts']} Zufallsstarts der Gewichtsiteration: {r['distinct']} verschiedene Enden, {V.pct(r['on_data'], 0)} % davon auf einem Kunden, {V.pct(r['best_share'], 0)} % in der besten gefundenen Mitte (innerhalb 0,2 km); "
                f"das schlechteste Ende kostet {V.de(r['worst_excess'], 1)} % mehr.")

with st.expander("🧰 Die Löser und die drei Kriterien erklärt"):
    st.markdown(
        r"""
| Verfahren | Was es tut | Was schiefgehen kann |
|---|---|---|
| **Newton** | Gedämpftes Newton-Verfahren (Armijo-Rückverfolgung) auf $\sum \|x_i - c\|^p$, für $p > 1$ | Braucht Rechnen in normierten Koordinaten, sonst Überlauf bei großem $p$; findet bei $p < 1$ nichts (nicht konvex). |
| **Gewichtsiteration ohne Dämpfung** | Neuer Punkt = Schwerpunkt mit Gewichten $w_i = d_i^{p-2}$ ($d_i$ = Abstand zu Kunde $i$) | Gemessen: konvergiert für $p \le 2{,}5$, bei $p = 3$ nur mit Vorbehalt, bei $p \ge 4$ nie (E7). Bei $p = 1$ ist das Weiszfeld-Verfahren. |
| **Gewichtsiteration mit Dämpfung** | Wie oben, aber nur der Bruchteil $1/(p-1)$ des Schritts | Gemessen: konvergiert in allen Läufen, braucht mehr Schritte als Newton. |
| **Vardi–Zhang** ($p = 1$) | Weiszfeld-Schritt mit Behandlung des Kunden, auf dem der Iterat steht (Stück 5 der Reihe) | Langsam nahe der Schwelle zwischen „Mitte zwischen den Kunden“ und „Mitte auf einem Kunden“. |
| **Kleinster umschließender Kreis** ($p = \infty$) | Inkrementelle Fassung nach Welzl: Kreis über zwei oder drei Kunden | Eindeutig; hängt allein von den äußersten Kunden ab. |

| Kriterium | Gemessen wird | Optimale Mitte |
|---|---|---|
| **Summe der Wege** ($q = 1$) | $\sum_i d_i$ | geometrischer Median ($p = 1$) |
| **Quadratsumme** ($q = 2$) | $\sum_i d_i^2$ | Schwerpunkt ($p = 2$) |
| **Längster Weg** ($q = \infty$) | $\max_i d_i$ | Mitte des kleinsten umschließenden Kreises ($p = \infty$) |

Die Dämpfung $1/(p-1)$ und die Begründung, warum die Gewichtsiteration für $p \le 2$ den Zielwert nie erhöht, sind **eigene Herleitungen** (siehe 📐), keine Quellenaussagen.
        """
    )

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.caption(f"Die Messreihen der Experimente sind vorgerechnet (vier feste Zufalls-Seeds je Zelle, Stand {R.meta()['stand']}); die Knöpfe „Nachmessen“ rechnen mit Ihren Einstellungen neu.")

# E1 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E1: Der Pfad der Mitte")
st.markdown("**Zahlenbeispiel auf der Achse.** Die Werte 1 bis 9 und 40, exakt gerechnet. Bei zehn Werten ist die Mitte für p = 1 kein Punkt, sondern das Intervall der Mediane (die Bilder legen sie in die Intervallmitte).")
st.plotly_chart(build_example_path(R.example()["rows"]), width="stretch", key="e1_example")
_verdict(V.verdict_e1_example(R.example()["rows"]))
st.markdown(f"**Stetig, aber nicht monoton.** „Strikt“ heißt: Die Mitte läuft von p = 1 bis p = ∞ nie gegen ihre Gesamtrichtung. Mittel über vier Zufalls-Seeds, {R.axis_cells()[0]['runs_per_seed']} Instanzen je Seed und Zelle.")
_fmt_table([{"Layout": c["layout"], "n": c["n"], "strikt (%)": 100 * c["strict"][0], "strikt bis p = 8 (%)": 100 * c["strict_to_8"][0], "Rückschritt im Mittel (% der Spannweite)": 100 * c["back_mean"],
             "größter Rückschritt (% der Spannweite)": 100 * c["back_max"]} for c in R.axis_cells()],
           **{"strikt (%)": "%.0f %%", "strikt bis p = 8 (%)": "%.0f %%", "Rückschritt im Mittel (% der Spannweite)": "%.2f %%", "größter Rückschritt (% der Spannweite)": "%.1f %%"})
_verdict(V.verdict_e1_axis(R.axis_cell("Normal", 51), R.axis_cell("Lognormal", 51), R.axis_cell("Zwei Häufungen", 51)))
st.markdown(f"**In der Ebene.** Wächst die Summe der Wege mit p, fällt der längste Weg (je Instanz zwischen p = 1 und p = ∞, {len(X.PLANE_P)} Stützstellen, {R.plane_cells()[0]['runs_per_seed']} Instanzen je Seed)?")
_fmt_table([{"Layout": c["layout"], "n": c["n"], "Summe wächst strikt (%)": 100 * c["sum_strict"][0], "längster Weg fällt strikt (%)": 100 * c["long_strict"][0],
             "größter Rückschritt Summe (%)": c["sum_back_max_pct"], "größter Rückschritt längster Weg (%)": c["long_back_max_pct"]} for c in R.plane_cells()],
           **{"Summe wächst strikt (%)": "%.0f %%", "längster Weg fällt strikt (%)": "%.0f %%", "größter Rückschritt Summe (%)": "%.1f %%", "größter Rückschritt längster Weg (%)": "%.1f %%"})
_verdict(V.verdict_e1_plane(R.plane_cell("Normal", 30), R.plane_cell("Außenposten", 30)))
st.markdown(f"**Wie groß muss p sein?** Abstand von c(p) zum 1-Center, in Radien des kleinsten umschließenden Kreises (n = {R.approach_cell('Normal')['n']}, {R.approach_cell('Normal')['runs_per_seed']} Instanzen je Seed).")
st.plotly_chart(build_approach_chart(R.load()["m1b"]["approach"]), width="stretch", key="e1_approach")
_verdict(V.verdict_e1_approach(R.load()["m1b"]["approach"]))
st.markdown("**Eigener Pfad auf der Achse.** Der Pfad für p von 0,1 bis ∞ auf Ihren eigenen Werten (senkrecht: p, logarithmisch). Für p < 1 springt die Mitte von Wert zu Wert.")
c1, c2, c3 = st.columns(3)
axis_kind = c1.selectbox("Werte", (EXAMPLE_VALUES,) + AXIS_LAYOUTS, key="axis_kind")
axis_n = c2.slider("Anzahl der Werte", 5, 201, 51, key="axis_n", disabled=axis_kind == EXAMPLE_VALUES)
axis_seed = c3.number_input("Zufalls-Seed der Werte", 0, C.SEED_MAX, 7, key="axis_seed")
xs = _axis_values(axis_kind, int(axis_n), int(axis_seed))
intervals = [K.solve_1d(xs, q) for q in AXIS_LIVE_P]
st.plotly_chart(build_axis_path(xs, AXIS_LIVE_P, intervals), width="stretch", key="axis_chart")

# E2 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E2: Die Kostenmatrix")
st.markdown("Wer die Mitte nach dem Exponenten p wählt und nach dem Kriterium q bewertet, zahlt gegenüber dem besten Punkt für q Mehrkosten (**Regret**, Prozent). Zeile: gewählt nach p ∈ {1, 2, ∞}; Spalte: bewertet nach q ∈ {1, 2, ∞}. "
            f"Mittel über vier Zufalls-Seeds, {R.regret_cell('Normal', 30)['runs_per_seed']} Instanzen je Seed (n = 300: {R.regret_cell('Normal', 300)['runs_per_seed']}).")
m1, m2 = st.columns(2)
e2_layout = m1.selectbox("Layout", C.LAYOUTS, key="e2_layout", index=2)
e2_n = m2.select_slider("n", X.N_RASTER, value=30, key="e2_n")
cell = R.regret_cell(e2_layout, int(e2_n))
st.plotly_chart(build_regret_heatmap(cell["mean"], f"{e2_layout}, n = {int(e2_n)}"), width="stretch", key="e2_heat")
st.caption(f"Streuung zwischen den Seeds: höchstens {V.de(max(max(r) for r in cell['std']), 1)} Prozentpunkte.")
st.plotly_chart(build_regret_by_n(R.regret_cells(), 1, 2, "Regret p = ∞, bewertet nach Quadratsumme (%)"), width="stretch", key="e2_by_n")
_verdict(V.verdict_e2(R.regret_cell("Außenposten", 30), R.regret_cell("Normal", 30)))
live_runs = X.live_runs(int(n))
if st.button(f"Nachmessen mit Layout „{layout}“, n = {int(n)} ({live_runs} Instanzen je Seed)", key="e2_live"):
    with st.spinner("Rechne vier Seeds …"):
        st.session_state["e2_result"] = (repr(settings), _regret_live(layout, int(n), live_runs, float(outpost), share / 100.0, float(sep), float(spread)))
res = st.session_state.get("e2_result")
if res is not None and res[0] == repr(settings):
    st.plotly_chart(build_regret_heatmap(res[1]["mean"], f"Nachgemessen: {layout}, n = {int(n)}"), width="stretch", key="e2_live_heat")

# E3 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E3: Der Außenposten")
st.markdown(f"{R.outpost_rows()[0]['n']} Kunden, einer davon bei (a; a), die übrigen {R.outpost_rows()[0]['n'] - 1} normalverteilt; {R.outpost_rows()[0]['runs_per_seed']} Instanzen je Seed. Abstand des Außenpostens vom Kern: a·√2.")
st.plotly_chart(build_outpost_chart(R.outpost_rows()), width="stretch", key="e3_chart")
_verdict(V.verdict_e3(R.outpost_rows()))

# E4 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E4: Das Kompromiss-p")
st.markdown("Gesucht ist das **eine feste p**, das über alle drei Kriterien am besten abschneidet: je p der größte Regret über q = 1, 2, ∞, gemittelt über Instanzen. Der Stern markiert das beste p des Rasters "
            "{1; 1,5; 2; 3; 4; 6; 8; 16; ∞}. Zum Vergleich: „je Instanz frei“ wählt für jede Instanz das beste p (braucht Kenntnis der Instanz).")
e4_n = st.select_slider("n", X.COMPROMISE_N, value=30, key="e4_n")
st.plotly_chart(build_compromise_chart(R.compromise_cell("Normal", int(e4_n)), [R.compromise_cell(lay, int(e4_n)) for lay in C.LAYOUTS[1:]]), width="stretch", key="e4_chart")
_fmt_table([{"Layout": c["layout"], "bestes festes p": V.fmt_p(V.unjp(c["best_p"])), "größter Regret dort (%)": c["best_value"], "je Instanz frei (%)": c["free_choice"],
             "bei p = 1 (%)": c["mean"][0], "bei p = 2 (%)": c["mean"][list(c["grid"]).index(2.0)], "bei p = ∞ (%)": c["mean"][-1]}
            for c in [R.compromise_cell(lay, int(e4_n)) for lay in C.LAYOUTS]],
           **{k: "%.1f %%" for k in ("größter Regret dort (%)", "je Instanz frei (%)", "bei p = 1 (%)", "bei p = 2 (%)", "bei p = ∞ (%)")})
_verdict(V.verdict_e4([R.compromise_cell(lay, 30) for lay in C.LAYOUTS]))
live_runs4 = X.live_runs(int(n), per_instance_cost=5.0)
if st.button(f"Nachmessen mit Layout „{layout}“, n = {int(n)} ({live_runs4} Instanzen je Seed)", key="e4_live"):
    with st.spinner("Rechne vier Seeds, neun Werte von p …"):
        st.session_state["e4_result"] = (repr(settings), _compromise_live(layout, int(n), live_runs4, float(outpost), share / 100.0, float(sep), float(spread)))
res = st.session_state.get("e4_result")
if res is not None and res[0] == repr(settings):
    cell4 = res[1]
    st.info(f"Nachgemessen ({layout}, n = {int(n)}): bestes festes p = **{V.fmt_p(V.unjp(cell4['best_p']))}** mit einem mittleren größten Regret von {V.de(cell4['best_value'], 1)} %; je Instanz frei gewählt {V.de(cell4['free_choice'], 1)} %.")
    st.plotly_chart(build_compromise_chart(cell4), width="stretch", key="e4_live_chart")

# E5 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E5: Die leere Zone")
st.markdown(f"Zwei Häufungen ({R.empty_plane_cells()[0]['n']} Kunden, Streuung 1 beide). „Leere Zone“: Der nächste Kunde ist weiter als {V.de(X.EMPTY_RADIUS_PLANE, 1)} km von der Mitte entfernt. {R.empty_plane_cells()[0]['runs_per_seed']} Instanzen je Seed.")
e5_sep = st.radio("Abstand der Häufungen", (6.0, 10.0), index=1, horizontal=True, key="e5_sep", format_func=lambda s: f"{s:g} km")
st.plotly_chart(build_empty_chart([c for c in R.empty_plane_cells() if c["sep"] == e5_sep]), width="stretch", key="e5_chart")
st.markdown(f"Auf der Achse ({R.empty_axis_cells()[0]['n']} Werte, N(0, 1) und N(8, 1); leere Zone: nächster Wert weiter als {V.de(X.EMPTY_RADIUS_AXIS, 1)}; {R.empty_axis_cells()[0]['runs_per_seed']} Instanzen je Seed):")
_fmt_table([{"Anteil der großen Häufung (%)": 100 * c["share"], **{f"p = {V.fmt_p(V.unjp(r['p']))} (%)": 100 * r["mean"] for r in c["rows"]}} for c in R.empty_axis_cells()],
           **{f"p = {V.fmt_p(q)} (%)": "%.0f %%" for q in (0.5, 1.2, 1.5, 2.0, 3.0, 4.0)})
_verdict(V.verdict_e5(R.empty_plane_cell(0.6, 10.0), next(c for c in R.empty_axis_cells() if c["share"] == 0.55)))

# E6 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E6: p kleiner als 1")
st.markdown(f"**Dichte gegen Masse (Achse).** {R.density_cells()[0]['n']} Werte: 60 % streuen mit s um 0, 40 % liegen dicht (Streuung 0,5) um 8. Anteil der Instanzen, in denen die Mitte in der kleinen, dichten Häufung liegt ({R.density_cells()[0]['runs_per_seed']} Instanzen je Seed).")
st.plotly_chart(build_density_chart(R.density_cells()), width="stretch", key="e6_density")
_verdict(V.verdict_e6_density(next(c for c in R.density_cells() if c["s_big"] == 3.0)))
st.markdown(f"**Der Pfad ist eine Treppe.** Größter Schritt der Mitte zwischen benachbarten p, geteilt durch die Spannweite ({R.stair_rows()[0]['n']} Werte, {R.stair_rows()[0]['runs_per_seed']} Instanzen je Seed). Für p ≥ 1 schrumpft er mit der Schrittweite (stetig), für p < 1 bleibt er (Sprünge von Kunde zu Kunde).")
_fmt_table([{"Layout": r["layout"], "Bereich von p": f"{r['lo']:g} bis {r['hi']:g}".replace(".", ","), "Schrittweite in p": r["step"], "größter Schritt (% der Spannweite)": 100 * r["largest_step"][0]} for r in R.stair_rows()],
           **{"Schrittweite in p": "%.3f", "größter Schritt (% der Spannweite)": "%.2f %%"})
st.markdown(f"**Zufallsstarts (Achse).** {R.start_rows()[0]['n']} Werte, 70 % N(0, 1) und 30 % N(8, 1), {R.start_rows()[0]['starts']} Zufallsstarts je Instanz, {R.start_rows()[0]['instances_per_seed']} Instanzen je Seed. B: Anteil der Enden innerhalb 0,2 des besten Minimums; A: Anteil in dessen Häufung.")
_fmt_table([{"p": r["p"], "B (%)": 100 * r["B"][0], "A (%)": 100 * r["A"][0], "Enden auf einem Kunden (%)": 100 * r["D"][0], "verschiedene Enden": r["distinct"][0]} for r in R.start_rows()],
           p="%.2f", **{"B (%)": "%.0f %%", "A (%)": "%.0f %%", "Enden auf einem Kunden (%)": "%.0f %%", "verschiedene Enden": "%.0f"})
_verdict(V.verdict_e6_start(R.start_rows()))
st.markdown(f"**In der Ebene.** Anteil der Instanzen, in denen das beste gefundene Minimum (alle Kunden und {C.STARTS_BELOW_ONE} Zufallsstarts) auf einem Kunden liegt ({R.plane_data_cells()[0]['runs_per_seed']} Instanzen je Seed).")
_fmt_table([{"Layout": c["layout"], "n": c["n"], **{f"p = {r['p']:g} (%)".replace(".", ","): 100 * r["mean"] for r in c["rows"]}} for c in R.plane_data_cells()],
           **{f"p = {q:g} (%)".replace(".", ","): "%.0f %%" for q in X.BELOW_ONE_P})
_verdict(V.verdict_e6_plane([c for c in R.plane_data_cells() if c["layout"] == "Normal"]))
st.markdown(f"**Dichte in der Ebene.** {R.plane_density_cells()[0]['n']} Kunden, 60 % in der großen Häufung mit Streuung s um (0; 0), 40 % mit Streuung 1 um (8; 0); Anteil der Instanzen, in denen die Mitte näher an der kleinen Häufung liegt (x > 4), {R.plane_density_cells()[0]['runs_per_seed']} Instanzen je Seed.")
_fmt_table([{"Streuung s": c["spread"], **{f"p = {r['p']:g} (%)".replace(".", ","): 100 * r["mean"] for r in c["rows"]}} for c in R.plane_density_cells()],
           **{f"p = {q:g} (%)".replace(".", ","): "%.0f %%" for q in (2.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2)})

# E7 ---------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 E7: Die Gewichtsiteration")
st.markdown(f"30 Kunden, Start im Schwerpunkt, Abbruch bei Schritt < 1e-10, Grenze {R.irls_cell('Normal')['maxit']} Schritte; {R.irls_cell('Normal')['runs_per_seed'] * 4} Instanzen je Layout und p. "
            "„Konvergiert“: Schritt unter der Toleranz vor der Grenze. „Zielwert steigt“: Der Zielwert Σ d^p nimmt bei mindestens einem Schritt zu.")
e7_layout = st.radio("Layout", ("Normal", "Zwei Häufungen"), horizontal=True, key="e7_layout")
cell7 = R.irls_cell(e7_layout)
st.plotly_chart(build_iterations_chart(cell7), width="stretch", key="e7_chart")
_fmt_table([{"p": r["p"], "ohne Dämpfung: konvergiert (%)": 100 * r["plain"]["converged"], "ohne Dämpfung: Zielwert steigt (%)": 100 * r["plain"]["rises"],
             "ohne Dämpfung: Schritte (Median)": r["plain"]["median_its"], "mit Dämpfung: konvergiert (%)": 100 * r["damped"]["converged"],
             "mit Dämpfung: Schritte (Median)": r["damped"]["median_its"], "mit Dämpfung: Schritte (größter Wert)": r["damped"]["max_its"], "Newton: Schritte (Median)": r["newton_median_its"]} for r in cell7["rows"]],
           p="%.1f", **{"ohne Dämpfung: konvergiert (%)": "%.0f %%", "ohne Dämpfung: Zielwert steigt (%)": "%.0f %%", "ohne Dämpfung: Schritte (Median)": "%.0f", "mit Dämpfung: konvergiert (%)": "%.0f %%",
                         "mit Dämpfung: Schritte (Median)": "%.0f", "mit Dämpfung: Schritte (größter Wert)": "%.0f", "Newton: Schritte (Median)": "%.0f"})
_verdict(V.verdict_e7(R.irls_cell("Normal"), R.irls_cell("Zwei Häufungen")))
if st.button("Die vier Werte p = 3, 4, 6, 8 auf Ihrer Instanz rechnen", key="e7_live"):
    with st.spinner("Rechne drei Löser …"):
        st.session_state["e7_result"] = (repr(settings), _irls_live(settings))
res = st.session_state.get("e7_result")
if res is not None and res[0] == repr(settings):
    _fmt_table([{"p": r["p"], "ohne Dämpfung": f"{r['plain']['its']} Schritte, {r['plain']['status']}", "ohne Dämpfung: Zielwert steigt": "ja" if r["plain"]["rises"] else "nein",
                 "mit Dämpfung": f"{r['damped']['its']} Schritte, {r['damped']['status']}", "Newton": f"{r['newton_its']} Schritte, {r['newton_status']}"} for r in res[1]], p="%.0f")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Luftlinie in der Ebene** | Straßen, Flüsse und Sperrungen fehlen; die Mitte kann in einem See oder auf einem Feld liegen. | Standortplanung mit diskreten Kandidaten |
| **Gleiche Gewichte** | Kunden mit verschiedener Menge verschieben jede Mitte; der Löser für p = 1 mit Gewichten ist Stück 5 der Reihe. | Stück 5 der Reihe (Weiszfeld) |
| **Ein einziges Lager** | Mehrere Mitten (k-Means, p-Center) sind ein anderes Problem; hier gibt es genau eine. | `kmeans-demo`, `p-center-demo` |
| **Keine Kapazität, keine Fixkosten** | Ein Lager hat Kosten und Grenzen; Summe, Quadratsumme und längster Weg sind nur Stellvertreter. | Kapazitierte Standortplanung |
| **Simulierte Layouts** | Alle Häufigkeiten und Raten stammen aus den vier Layouts dieser Demo; echte Kundenlagen können davon stark abweichen. | – |
| **Kompromiss-p nur für ein Raster** | Das beste p stammt aus neun Werten {1; 1,5; 2; 3; 4; 6; 8; 16; ∞}, vier Layouts und n = 10, 30, 100; dazwischen und für andere Layouts ist es nicht untersucht. | – |
| **p < 1 mit @@STARTS@@ Zufallsstarts** | In der Ebene werden alle Kunden und @@STARTS@@ Zufallsstarts verglichen: das beste von endlich vielen Kandidaten, kein Beweis des globalen Minimums. | – |
| **Konvergenz nur gemessen** | „Gewichtsiteration konvergiert ab p = 4 nicht“ ist eine Messung an @@IRLS@@ Instanzen je Layout und p, kein Beweis. | – |
| **Streuung zwischen den Seeds** | Jede Zelle nutzt vier feste Seeds; die angegebene Streuung ist die zwischen diesen Seeds, kein Konfidenzintervall. | – |
| **Exponent und Norm** | p ist der Exponent der Strecke, nicht die Norm der Strecke; die Strecke bleibt die euklidische Luftlinie. | – |
    """.replace("@@IRLS@@", str(R.irls_cell("Normal")["runs_per_seed"] * len(R.meta()["seeds"]))).replace("@@STARTS@@", str(C.STARTS_BELOW_ONE))
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Kunden $x_1, \dots, x_n$ in der Ebene (gleiche Gewichte). Gesucht ist die Mitte $c(p)$ zum Exponenten $p$.

| Symbol | Bedeutung |
|---|---|
| $x_i$ | Ort von Kunde $i$ |
| $c$ | Ort des Lagers (die Mitte) |
| $d_i = \|x_i - c\|$ | Luftlinie von Kunde $i$ zum Lager |
| $p$ | **Exponent der Strecke** (nicht die Zahl der Mittelpunkte wie in der p-center-demo) |
| $q$ | Exponent des Bewertungskriteriums (1, 2 oder $\infty$) |
| $w_i$ | Gewicht in der Gewichtsiteration |
| $R_{p \to q}$ | Regret: Mehrkosten, wenn nach $p$ gewählt und nach $q$ bewertet wird |

**Die Lp-Mitte.**

$$c(p) = \arg\min_c \sum_{i=1}^{n} \|x_i - c\|^{p}.$$

**Drei bekannte Stellen.** Für $p = 2$ ist die Mitte der Schwerpunkt:

$$c(2) = \frac{1}{n}\sum_{i=1}^{n} x_i.$$

Für $p = 1$ ist sie der geometrische Median (Stück 5 der Reihe), und für $p \to \infty$ minimiert sie den längsten Weg:

$$c(\infty) = \arg\min_c \max_i \|x_i - c\|.$$

**Konvexität.** Für $p > 1$ ist $c \mapsto \sum_i \|x_i - c\|^p$ streng konvex, die Mitte ist eindeutig. Für $0 < p < 1$ ist die Summe zwischen zwei Kunden auf der Achse konkav, das Minimum liegt also auf einem Kunden; in der Ebene ist jeder Kunde ein lokales Minimum.

**Gewichtsiteration.** Mit $w_i = d_i^{\,p-2}$ ist

$$c^{+} = \frac{\sum_i w_i\, x_i}{\sum_i w_i}.$$

Für $p = 1$ ist das das Weiszfeld-Verfahren, für $p = 2$ der Schwerpunkt in einem Schritt.

**Dämpfung (eigene Herleitung, keine Quellenaussage).** Auf der Achse ist der Schritt der Gewichtsiteration das $(p-1)$-fache des Newton-Schritts bei festem Punkt:

$$\Delta_{\text{Gewichtsiteration}} = (p-1)\,\Delta_{\text{Newton}}.$$

Die gedämpfte Iteration geht deshalb nur den Bruchteil $1/(p-1)$ des Schritts:

$$c^{+} = c + \frac{1}{p-1}\left(\frac{\sum_i w_i\, x_i}{\sum_i w_i} - c\right).$$

In der Ebene ist die Hesse-Matrix je Kunde $p\,d_i^{p-2}\big(I + (p-2)\,u_i u_i^\top\big)$ mit $u_i = (c - x_i)/d_i$; ihre Eigenwerte liegen für $p > 2$ zwischen dem 1- und dem $(p-1)$-fachen des Werts $p\,d_i^{p-2}$, den die Gewichtsiteration ansetzt. Das erklärt, warum $1/(p-1)$ ein natürlicher Schritt ist, **beweist aber keine Konvergenz**; die Konvergenz ist hier nur gemessen (E7).

**Warum für $p \le 2$ nichts steigt (eigene Herleitung).** Für $p \le 2$ ist $t \mapsto t^{p/2}$ konkav, also liegt es unter seiner Tangente in $t_0 = d_0^2$:

$$d^{p} \le \frac{p}{2}\,d_0^{\,p-2}\,d^{2} + \left(1 - \frac{p}{2}\right) d_0^{\,p}.$$

Die rechte Seite ist eine gewichtete Quadratsumme, ihr Minimierer ist genau der Schritt der Gewichtsiteration; so kann der Zielwert nicht steigen. Für $p > 2$ ist $t^{p/2}$ konvex, die Tangente liegt darunter, und die Begründung entfällt. Gemessen wurde die Monotonie nur für $p = 1{,}5$ bis $2{,}5$ (E7).

**Regret.** Kosten zum Exponenten $q$:

$$K_q(c) = \sum_{i=1}^{n} \|x_i - c\|^{q}, \qquad K_\infty(c) = \max_i \|x_i - c\|.$$

Der Regret der Wahl nach $p$, bewertet nach $q$:

$$R_{p \to q} = \frac{K_q\big(c(p)\big)}{K_q\big(c(q)\big)} - 1.$$

**Kompromiss-p.** Das beste feste $p$ eines Rasters $\mathcal{P}$ minimiert den mittleren größten Regret:

$$p^{*} = \arg\min_{p \in \mathcal{P}}\ \operatorname{Mittel}_{\text{Instanzen}}\ \max_{q \in \{1,\,2,\,\infty\}} R_{p \to q}.$$

**Kleinster umschließender Kreis.** Er ist eindeutig und wird von höchstens drei Kunden bestimmt. Die Demo rechnet die inkrementelle Fassung nach der Beschreibung des Verfahrens (Welzl); die Originalarbeit wurde **nicht eingesehen**.

**Literatur.** Für die Gewichtsiteration und die Lp-Mitte wurde keine Quelle gelesen; die Aussagen dieser Demo beruhen auf eigener Herleitung und auf Messungen und sind gegen `scipy.optimize`, ein Gitter und die Aufzählung aller Paare und Tripel geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
