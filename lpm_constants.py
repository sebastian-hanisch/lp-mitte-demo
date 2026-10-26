"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Einheit der Ebene: 1 Längeneinheit = 1 km (Luftlinie). Alle Kunden haben das gleiche Gewicht (Gewichte sind ein anderes Stück der Reihe).

Namenskonvention (Buchstabe p): In dieser Demo ist **p der Exponent** der Strecke in Σ d^p. In der p-center-demo bedeutet p die Zahl der Mittelpunkte,
im Potenzmittel der Reihe heißt der Exponent anders. Das Bewertungskriterium trägt den Buchstaben q; der Anteil der großen Häufung heißt „Anteil“."""

import math

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
LAYOUTS = ("Normal", "Quadrat", "Außenposten", "Zwei Häufungen")
CRITERIA = ("Summe der Wege (q = 1)", "Quadratsumme (q = 2)", "längster Weg (q = ∞)")
CRITERION_Q = (1.0, 2.0, math.inf)
SOLVERS = ("Newton", "Gewichtsiteration ohne Dämpfung", "Gewichtsiteration mit Dämpfung")
STARTS = ("Schwerpunkt", "zufälliger Punkt")
P_OPTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, math.inf)
P_FINITE_MAX = 64.0

N_MIN, N_MAX, DEFAULT_N = 5, 300, 30               # Karte und Kosten: O(n); Gewichtsiteration/Newton je Schritt O(n); p < 1: O(n²) für die Datenpunkte
OUTPOST_MIN, OUTPOST_MAX, DEFAULT_OUTPOST = 0.0, 20.0, 6.0   # Außenposten bei (a, a)
SHARE_MIN, SHARE_MAX, DEFAULT_SHARE = 50, 95, 70   # Anteil der großen Häufung in Prozent
SEP_MIN, SEP_MAX, DEFAULT_SEP = 2.0, 12.0, 6.0     # Abstand der Häufungen in x-Richtung
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.5, 4.0, 1.0   # Streuung der großen Häufung (die kleine hat Streuung 1)
SEED_MAX, DEFAULT_SEED = 999999, 7
CENTER_P_DEFAULT = 2.0

# --- Rechenbudget ---------------------------------------------------------------------------------------------------------------------------
STARTS_BELOW_ONE = 20       # Zufallsstarts der Gewichtsiteration für p < 1 in der Ebene (live)
BELOW_ONE_LIVE_MAX_N = 100  # p < 1 in der Ebene: Matrix der Datenpunkte O(n²) und 20 Starts; darüber nicht live nachgemessen
PATH_P = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0)   # Stützstellen des Pfades über p ≥ 1 (∞ kommt dazu)
LIVE_BUDGET_INSTANCES = 400  # Nachmessen: Instanzen je Seed höchstens so viele, je nach n weniger (siehe lpm_experiments.live_runs)
IRLS_MAXIT_LIVE = 3000

# --- Messreihen (vorgerechnet in data/lpm_results.json, erzeugt von tools/make_results.py) -------------------------------------------------
SEEDS = (11, 12, 13, 14)    # vier feste Zufalls-Seeds je Zelle: nie ein einzelner Seed als Befund
RESULTS_FILE = "lpm_results.json"
GRID_P = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0, math.inf)   # Raster des Kompromiss-p
OUTPOST_RASTER = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0)

# --- Farben -------------------------------------------------------------------------------------------------------------------------------
COLOR_POINT = "rgba(80,80,80,0.55)"
COLOR_MEAN = "#7f7f7f"
COLOR_MEDIAN = "#2F6B65"
COLOR_CENTER = "#d62728"
COLOR_ONE_CENTER = "#9467bd"
COLOR_PATH = "#4c78a8"
COLOR_LINE = "#4c78a8"
COLOR_LINE2 = "#e07b00"

# --- Presets (Werte für jeden Regler; Seeds und Werte in tests/test_presets.py auf ihr Verhalten geprüft) -----------------------------------
_BASE = {
    "layout": "Normal", "n": DEFAULT_N, "outpost": DEFAULT_OUTPOST, "share": DEFAULT_SHARE, "sep": DEFAULT_SEP, "spread": DEFAULT_SPREAD,
    "p": CENTER_P_DEFAULT, "criterion": CRITERIA[0], "solver": SOLVERS[0], "start": STARTS[0], "seed": DEFAULT_SEED,
}


def _preset(**changes):
    unknown = set(changes) - set(_BASE)
    assert not unknown, unknown
    return {**_BASE, **changes}


PRESETS = {
    "Standard (Normal, n = 30, p = 2)": _preset(),
    "Ein Außenposten (a = 6)": _preset(layout="Außenposten", criterion=CRITERIA[2]),
    "Zwei Liefergebiete": _preset(layout="Zwei Häufungen", n=100, share=60, sep=10.0, seed=8),
    "Dichte gegen Masse": _preset(layout="Zwei Häufungen", n=100, share=60, sep=8.0, spread=3.0, p=0.4, seed=8),
    "Kompromiss-p": _preset(layout="Außenposten", p=3.0, criterion=CRITERIA[2]),
    "Die Iteration hängt": _preset(p=6.0, solver=SOLVERS[1]),
}
PRESET_HELP = {
    "Standard (Normal, n = 30, p = 2)": "30 normalverteilte Kunden, Exponent p = 2: Die Mitte ist der Schwerpunkt. Ändern Sie p und sehen Sie die Mitte auf dem Pfad wandern: p = 1 ist der geometrische Median, "
                                        "sehr große p nähern sich der Mitte des kleinsten umschließenden Kreises.",
    "Ein Außenposten (a = 6)": "29 normalverteilte Kunden und einer bei (6; 6). Der Schwerpunkt (p = 2) liegt beim Summe-Kriterium nur 1,4 % über dem Optimum, aber nach dem längsten Weg bewertet "
                               "57,9 % über dem 1-Center. Stellen Sie p auf 1 oder ∞ und vergleichen Sie die Kennzahlen.",
    "Zwei Liefergebiete": "100 Kunden, 60 % um (0; 0), 40 % um (10; 0). Der Schwerpunkt (p = 2) liegt bei x = 4,06 und damit 2,27 km vom nächsten Kunden entfernt in der leeren Zone zwischen den Gebieten; "
                          "der geometrische Median (p = 1) liegt bei x = 1,18 in der großen Häufung.",
    "Dichte gegen Masse": "100 Kunden, 60 % breit gestreut (Streuung 3) um (0; 0), 40 % dicht (Streuung 1) um (8; 0). Bei p = 1 liegt die Mitte bei x = 3,42, bei p = 0,4 springt sie auf einen Kunden der "
                          "dichten Häufung (x = 7,49), obwohl dort nur 40 % der Kunden stehen. Der einzelne Start der Gewichtsiteration findet dieses Minimum nicht: Er endet 5,3 km davon entfernt.",
    "Kompromiss-p": "Derselbe Außenposten-Fall mit p = 3: Der größte Regret über die drei Kriterien sinkt von 57,9 % (p = 2) auf 45,8 %; dafür kosten Summe und Quadratsumme 10,7 % und 9,2 % mehr. "
                    "Das beste feste p hängt vom Layout ab (E4).",
    "Die Iteration hängt": "p = 6 mit der Gewichtsiteration ohne Dämpfung: Nach 3000 Schritten ist die Grenze erreicht, der Lauf liegt rund 1,2 km neben der Mitte. "
                           "Mit Dämpfung (unter 50 Schritte) oder mit Newton (unter 10 Schritte) ist sie da.",
}
