"""Messreihen der Experimente E1 bis E7: reine Funktionen über feste Zufalls-Seeds, die sowohl die vorgerechneten Tabellen (tools/make_results.py → data/lpm_results.json)
als auch die Live-Knöpfe der App erzeugen. Jede Zelle nutzt vier feste Seeds (C.SEEDS); angegeben wird der Mittelwert mit der Standardabweichung über die Seeds.

Zufall: `Draws(cell_seed(...))` aus lpm_scenario.py; ein Seed gehört zu einer Zelle (Seed, Marken), nie zu einer Instanz allein.
Unendlich wird in den Ergebnissen als Text "inf" gespeichert (gültiges JSON), siehe `jp`."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

import lpm_center as K
import lpm_constants as C
import lpm_regret as RG
from lpm_scenario import AXIS_LAYOUTS, Draws, axis_batch, cell_seed, density_batch, layout_batch

ROOT = Path(__file__).resolve().parent
SOURCES = ("lpm_rng.py", "lpm_scenario.py", "lpm_geomed.py", "lpm_center.py", "lpm_regret.py", "lpm_experiments.py", "tools/make_results.py")
INF = math.inf

AXIS_P = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, INF)
PLANE_P = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, INF)
APPROACH_P = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
EXAMPLE_P = (1.0, 1.1, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, INF)
N_RASTER = (10, 30, 100, 300)
COMPROMISE_N = (10, 30, 100)
MONOTONE_N_AXIS = (9, 51, 201)
MONOTONE_N_PLANE = (30, 100)
EMPTY_RADIUS_PLANE = 1.5
EMPTY_RADIUS_AXIS = 1.0
IRLS_P = (1.5, 2.5, 3.0, 4.0, 6.0, 8.0)
IRLS_MAXIT = 5000
IRLS_N = 30


def jp(p: float):
    """p für JSON: ∞ als Text 'inf'."""
    return "inf" if p == INF else float(p)


def unjp(v) -> float:
    return INF if v == "inf" else float(v)


def source_hash() -> str:
    """Prüfsumme der Quelldateien der Messreihen (Zeilenenden normalisiert): stimmt sie nicht mit der in der JSON-Datei überein, ist die Datei veraltet."""
    h = hashlib.sha256()
    for name in SOURCES:
        text = (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(name.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()[:16]


def mean_std(values) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std())


def live_runs(n: int, per_instance_cost: float = 1.0) -> int:
    """Instanzen je Seed für das Nachmessen in der App: so viele, dass der Aufwand (Instanzen · n) unter dem Budget bleibt, mindestens 20."""
    return int(max(20, min(C.LIVE_BUDGET_INSTANCES, 40000 // max(n, 1) / per_instance_cost)))


# ---------------------------------------------------------------- E1: der Pfad c(p)

def number_example() -> dict:
    """Das Zahlenbeispiel auf der Achse: die Werte 1 bis 9 und 40, c(p) als Intervall (für p = 1 der Bereich der Mediane)."""
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 40], dtype=float)
    rows = []
    for p in EXAMPLE_P:
        lo, hi = K.solve_1d(x, p)
        rows.append({"p": jp(p), "lo": lo, "hi": hi})
    return {"values": x.tolist(), "rows": rows}


def reversal(path, direction: int, scale: float) -> tuple[bool, float]:
    """Gegen die Richtung `direction` (+1 wachsend, −1 fallend) laufende Schritte des Pfades: (nie gegen die Richtung, größter Rückschritt geteilt durch `scale`)."""
    back = np.maximum(0.0, -direction * np.diff(np.asarray(path, dtype=float)))
    worst = float(back.max())
    return worst <= 1e-12 * scale, worst / scale


def axis_monotone_cell(layout: str, n: int, runs: int, seeds=C.SEEDS) -> dict:
    """Pfad c(p) auf der Achse (ungerades n: für p = 1 ein Punkt): Anteil der Instanzen, deren Pfad nie gegen seine Richtung läuft ('strikt'), und der größte Rückschritt in Spannweiten.
    Richtung = Vorzeichen von c(∞) − c(1). Zweite Zahl: dasselbe nur bis p = 8."""
    strict, strict8, back_mean, back_p95, back_max = [], [], [], [], []
    for seed in seeds:
        xs = axis_batch(Draws(cell_seed(seed, "axis", layout, n)), layout, runs, n)
        flags, flags8, backs = [], [], []
        for x in xs:
            span = float(x.max() - x.min())
            path = np.array([K.solve_1d(x, p)[0] for p in AXIS_P])
            direction = 1 if path[-1] >= path[0] else -1
            s, b = reversal(path, direction, span)
            s8, _ = reversal(path[:AXIS_P.index(8.0) + 1], direction, span)
            flags.append(s)
            flags8.append(s8)
            backs.append(b)
        strict.append(np.mean(flags))
        strict8.append(np.mean(flags8))
        back_mean.append(np.mean(backs))
        back_p95.append(np.percentile(backs, 95))
        back_max.append(np.max(backs))
    return {"layout": layout, "n": n, "runs_per_seed": runs, "strict": mean_std(strict), "strict_to_8": mean_std(strict8),
            "back_mean": mean_std(back_mean)[0], "back_p95": mean_std(back_p95)[0], "back_max": float(max(back_max))}


def plane_path(points: np.ndarray, ps, seed: int = 0) -> list[np.ndarray]:
    """Die Mitten c(p) für die Exponenten ps (alle ≥ 1)."""
    return [K.lp_center(points, p, seed) for p in ps]


def plane_monotone_cell(layout: str, n: int, runs: int, seeds=C.SEEDS) -> dict:
    """Pfad in der Ebene: wächst die Summe der Wege mit p (strikt, nie zurück), fällt der längste Weg? Anteil der Instanzen und größter Rückschritt in Prozent des Wertes."""
    sum_strict, long_strict, sum_back, long_back = [], [], [], []
    for seed in seeds:
        pts_all = layout_batch(Draws(cell_seed(seed, "plane", layout, n)), layout, runs, n)
        a, b, ra, rb = [], [], [], []
        for pts in pts_all:
            centers = plane_path(pts, PLANE_P)
            sums = [K.lp_cost(pts, c, 1.0) for c in centers]
            longest = [K.lp_cost(pts, c, INF) for c in centers]
            s1, r1 = reversal(sums, +1, sums[0])
            s2, r2 = reversal(longest, -1, longest[0])
            a.append(s1)
            b.append(s2)
            ra.append(r1)
            rb.append(r2)
        sum_strict.append(np.mean(a))
        long_strict.append(np.mean(b))
        sum_back.append(max(ra))
        long_back.append(max(rb))
    return {"layout": layout, "n": n, "runs_per_seed": runs, "sum_strict": mean_std(sum_strict), "long_strict": mean_std(long_strict),
            "sum_back_max_pct": 100 * float(max(sum_back)), "long_back_max_pct": 100 * float(max(long_back))}


def approach_cell(layout: str, n: int, runs: int, seeds=C.SEEDS) -> dict:
    """Wie groß muss p sein? Je p: Abstand von c(p) zum 1-Center in Radien, Überschreitung des längsten Wegs gegen den Radius und Mehrweg der Summe gegen den geometrischen Median (Mittel über Instanzen)."""
    dist = {p: [] for p in APPROACH_P}
    over = {p: [] for p in APPROACH_P}
    extra = {p: [] for p in APPROACH_P}
    for seed in seeds:
        pts_all = layout_batch(Draws(cell_seed(seed, "approach", layout, n)), layout, runs, n)
        acc = {p: ([], [], []) for p in APPROACH_P}
        for pts in pts_all:
            c1 = K.lp_center(pts, 1.0)
            cinf, radius = K.one_center(pts)
            s1 = K.lp_cost(pts, c1, 1.0)
            for p in APPROACH_P:
                c = K.newton_lp(pts, p).center
                acc[p][0].append(np.linalg.norm(c - cinf) / radius)
                acc[p][1].append(100 * (K.lp_cost(pts, c, INF) / radius - 1))
                acc[p][2].append(100 * (K.lp_cost(pts, c, 1.0) / s1 - 1))
        for p in APPROACH_P:
            dist[p].append(np.mean(acc[p][0]))
            over[p].append(np.mean(acc[p][1]))
            extra[p].append(np.mean(acc[p][2]))
    return {"layout": layout, "n": n, "runs_per_seed": runs,
            "rows": [{"p": p, "dist_radius": mean_std(dist[p])[0], "overshoot_pct": mean_std(over[p])[0], "sum_excess_pct": mean_std(extra[p])[0]} for p in APPROACH_P]}


def run_m1a() -> dict:
    cells = [axis_monotone_cell(layout, n, 125) for layout in AXIS_LAYOUTS for n in MONOTONE_N_AXIS]
    return {"example": number_example(), "axis": cells}


def run_m1b() -> dict:
    plane = [plane_monotone_cell(layout, n, 50) for layout in C.LAYOUTS for n in MONOTONE_N_PLANE]
    approach = [approach_cell(layout, 30, 100) for layout in C.LAYOUTS]
    return {"plane": plane, "approach": approach}


# ---------------------------------------------------------------- E2/E3: die Kostenmatrix, der Außenposten

def regret_cell(layout: str, n: int, runs: int, seeds=C.SEEDS, **kw) -> dict:
    """Mittlere Regret-Matrix (3×3, Prozent; Zeile p = 1, 2, ∞, Spalte q = 1, 2, ∞) über `runs` Instanzen je Seed; Streuung zwischen den Seeds."""
    per_seed = []
    for seed in seeds:
        pts_all = layout_batch(Draws(cell_seed(seed, "regret", layout, n, *kw.values())), layout, runs, n, **kw)
        per_seed.append(np.mean([RG.regret_matrix(pts) for pts in pts_all], axis=0))
    stack = np.array(per_seed)
    return {"layout": layout, "n": n, "runs_per_seed": runs, "mean": stack.mean(axis=0).tolist(), "std": stack.std(axis=0).tolist()}


def run_m2() -> dict:
    raster = [regret_cell(layout, n, 200 if n <= 100 else 100) for layout in C.LAYOUTS for n in N_RASTER]
    outpost = []
    for a in C.OUTPOST_RASTER:
        cell = regret_cell("Außenposten", 31, 150, outpost=a)
        outpost.append({"a": a, **{k: cell[k] for k in ("n", "runs_per_seed", "mean", "std")}})
    return {"raster": raster, "outpost": outpost}


# ---------------------------------------------------------------- E4: das Kompromiss-p

def compromise_cell(layout: str, n: int, runs: int, seeds=C.SEEDS, grid=C.GRID_P, **kw) -> dict:
    """Je p des Rasters der mittlere größte Regret (über die drei Kriterien), das beste feste p, und der Wert, wenn je Instanz das beste p gewählt werden dürfte."""
    per_seed, free = [], []
    for seed in seeds:
        pts_all = layout_batch(Draws(cell_seed(seed, "compromise", layout, n, *kw.values())), layout, runs, n, **kw)
        table = np.array([RG.compromise_regrets(pts, grid) for pts in pts_all])      # (runs, len(grid))
        per_seed.append(table.mean(axis=0))
        free.append(table.min(axis=1).mean())
    stack = np.array(per_seed)
    mean = stack.mean(axis=0)
    best = int(np.argmin(mean))
    return {"layout": layout, "n": n, "runs_per_seed": runs, "grid": [jp(p) for p in grid], "mean": mean.tolist(), "std": stack.std(axis=0).tolist(),
            "best_p": jp(grid[best]), "best_value": float(mean[best]), "free_choice": mean_std(free)[0]}


def run_m4() -> dict:
    return {"cells": [compromise_cell(layout, n, 150 if n <= 30 else 80) for layout in C.LAYOUTS for n in COMPROMISE_N]}


# ---------------------------------------------------------------- E5: die leere Zone

def empty_plane_cell(share: float, sep: float, n: int, runs: int, seeds=C.SEEDS) -> dict:
    """Anteil der Instanzen, in denen die Mitte (p = 1, 2, ∞) in einer leeren Zone liegt (nächster Kunde weiter als EMPTY_RADIUS_PLANE)."""
    per_seed = {q: [] for q in RG.Q_VALUES}
    for seed in seeds:
        hits = {q: 0 for q in RG.Q_VALUES}
        for pts in layout_batch(Draws(cell_seed(seed, "empty", share, sep, n)), "Zwei Häufungen", runs, n, share=share, sep=sep):
            for q, c in RG.optimal_centers(pts).items():
                hits[q] += RG.empty_zone(pts, c, EMPTY_RADIUS_PLANE)
        for q in RG.Q_VALUES:
            per_seed[q].append(hits[q] / runs)
    return {"share": share, "sep": sep, "n": n, "runs_per_seed": runs, "rows": [{"p": jp(q), "mean": mean_std(per_seed[q])[0], "std": mean_std(per_seed[q])[1]} for q in RG.Q_VALUES]}


def empty_axis_cell(share: float, n: int, runs: int, ps, seeds=C.SEEDS) -> dict:
    """Auf der Achse: zwei Häufungen N(0, 1) und N(8, 1); Anteil der Instanzen, in denen c(p) weiter als EMPTY_RADIUS_AXIS vom nächsten Wert liegt."""
    rows = []
    per_seed = {p: [] for p in ps}
    for seed in seeds:
        hits = {p: 0 for p in ps}
        for x in axis_batch(Draws(cell_seed(seed, "emptyaxis", share, n)), "Zwei Häufungen", runs, n, share=share, sep=8.0):
            for p in ps:
                c = K.solve_1d(x, p)[0]
                hits[p] += bool(np.abs(x - c).min() > EMPTY_RADIUS_AXIS)
        for p in ps:
            per_seed[p].append(hits[p] / runs)
    for p in ps:
        rows.append({"p": jp(p), "mean": mean_std(per_seed[p])[0], "std": mean_std(per_seed[p])[1]})
    return {"share": share, "n": n, "runs_per_seed": runs, "rows": rows}


def run_m5() -> dict:
    plane = [empty_plane_cell(share, sep, 100, 150) for sep in (6.0, 10.0) for share in (0.5, 0.6, 0.7, 0.8)]
    axis = [empty_axis_cell(share, 200, 100, (0.5, 1.2, 1.5, 2.0, 3.0, 4.0)) for share in (0.5, 0.55, 0.6, 0.7, 0.8)]
    return {"plane": plane, "axis": axis}


# ---------------------------------------------------------------- E6: p kleiner als 1

DENSITY_P = (2.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1)
START_P = (2.0, 0.95, 0.9, 0.7, 0.5, 0.2)
BELOW_ONE_P = (0.9, 0.7, 0.5, 0.3, 0.1)


def density_axis_cell(s_big: float, n: int, runs: int, seeds=C.SEEDS) -> dict:
    """Achse: 60 % der Werte um 0 mit Streuung s_big, 40 % dicht um 8 (Streuung 0,5). Anteil der Instanzen, in denen c(p) näher an 8 als an 0 liegt (in der dichten kleinen Häufung)."""
    per_seed = {p: [] for p in DENSITY_P}
    for seed in seeds:
        hits = {p: 0 for p in DENSITY_P}
        for x in density_batch(Draws(cell_seed(seed, "density", s_big, n)), runs, n, s_big):
            for p in DENSITY_P:
                lo, hi = K.solve_1d(x, p)
                hits[p] += (lo + hi) / 2 > 4.0
        for p in DENSITY_P:
            per_seed[p].append(hits[p] / runs)
    return {"s_big": s_big, "n": n, "runs_per_seed": runs, "rows": [{"p": p, "mean": mean_std(per_seed[p])[0], "std": mean_std(per_seed[p])[1]} for p in DENSITY_P]}


def axis_start_study(p: float, instances: int, n: int = 200, starts: int = 200, seeds=C.SEEDS) -> dict:
    """Achse, 70 % N(0, 1) und 30 % N(8, 1): `starts` Zufallsstarts der Gewichtsiteration je Instanz.
    B: Anteil der Enden innerhalb 0,2 des globalen Minimums; A: Anteil in der Häufung des globalen Minimums; D: Anteil der Enden auf einem Datenpunkt;
    distinct: Zahl verschiedener Enden; conv: Anteil der Starts, die die Toleranz erreichen. Mittel über Instanzen, dann über Seeds."""
    keys = ("B", "A", "D", "distinct", "conv", "glob")
    per_seed = {k: [] for k in keys}
    for seed in seeds:
        draws = Draws(cell_seed(seed, "start", p))
        acc = {k: [] for k in keys}
        for x in axis_batch(draws, "Zwei Häufungen", instances, n, share=0.7, sep=8.0):
            glob = K.solve_1d(x, p)[0]
            s = x.min() + draws.uniform(starts) * (x.max() - x.min())
            ends, its = K.irls_multi(x[:, None], p, s[:, None], maxit=C.IRLS_MAXIT_LIVE)
            e = ends[:, 0]
            acc["B"].append(np.mean(np.abs(e - glob) <= 0.2))
            acc["A"].append(np.mean((e > 4.0) == (glob > 4.0)))
            acc["D"].append(np.mean(np.abs(e[:, None] - x[None, :]).min(axis=1) < 1e-6))
            acc["distinct"].append(len(np.unique(np.round(e, 6))))
            acc["conv"].append(np.mean(its < C.IRLS_MAXIT_LIVE))
            acc["glob"].append(glob)
        for k in keys:
            per_seed[k].append(np.mean(acc[k]))
    return {"p": p, "instances_per_seed": instances, "n": n, "starts": starts, **{k: mean_std(per_seed[k]) for k in keys}}


def staircase_cell(layout: str, n: int, runs: int, lo: float, hi: float, step: float, seeds=C.SEEDS) -> dict:
    """Pfad auf der Achse über p von lo bis hi in Schritten `step`: der größte Schritt zwischen benachbarten p, geteilt durch die Spannweite (Mittel der Instanzen-Maxima, dann über Seeds)."""
    ps = np.arange(lo, hi + step / 2, step)
    per_seed = []
    for seed in seeds:
        biggest = []
        for x in axis_batch(Draws(cell_seed(seed, "stair", layout, n)), layout, runs, n):
            path = np.array([K.solve_1d(x, float(p))[0] for p in ps])
            biggest.append(np.abs(np.diff(path)).max() / (x.max() - x.min()))
        per_seed.append(np.mean(biggest))
    return {"layout": layout, "n": n, "lo": lo, "hi": hi, "step": step, "runs_per_seed": runs, "largest_step": mean_std(per_seed)}


def below_one_plane_cell(layout: str, n: int, runs: int, seeds=C.SEEDS, **kw) -> dict:
    """Ebene, 0 < p < 1: Anteil der Instanzen, in denen das beste gefundene Minimum auf einem Kunden liegt (best_below_one mit 20 Zufallsstarts)."""
    per_seed = {p: [] for p in BELOW_ONE_P}
    for seed in seeds:
        pts_all = layout_batch(Draws(cell_seed(seed, "below", layout, n)), layout, runs, n, **kw)
        hits = {p: 0 for p in BELOW_ONE_P}
        for i, pts in enumerate(pts_all):
            for p in BELOW_ONE_P:
                hits[p] += K.best_below_one(pts, p, seed=i, starts=C.STARTS_BELOW_ONE).on_data
        for p in BELOW_ONE_P:
            per_seed[p].append(hits[p] / runs)
    return {"layout": layout, "n": n, "runs_per_seed": runs, "rows": [{"p": p, "mean": mean_std(per_seed[p])[0], "std": mean_std(per_seed[p])[1]} for p in BELOW_ONE_P]}


def density_plane_cell(spread: float, n: int, runs: int, share: float = 0.6, sep: float = 8.0, seeds=C.SEEDS) -> dict:
    """Ebene, zwei Häufungen (große mit Streuung `spread`, kleine mit Streuung 1): Anteil der Instanzen, in denen die Mitte näher an der kleinen (dichteren) Häufung liegt (x > sep/2)."""
    ps = (2.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2)
    per_seed = {p: [] for p in ps}
    for seed in seeds:
        hits = {p: 0 for p in ps}
        for i, pts in enumerate(layout_batch(Draws(cell_seed(seed, "dplane", spread, n)), "Zwei Häufungen", runs, n, share=share, sep=sep, spread=spread)):
            for p in ps:
                hits[p] += K.lp_center(pts, p, i)[0] > sep / 2
        for p in ps:
            per_seed[p].append(hits[p] / runs)
    return {"spread": spread, "n": n, "share": share, "sep": sep, "runs_per_seed": runs, "rows": [{"p": p, "mean": mean_std(per_seed[p])[0], "std": mean_std(per_seed[p])[1]} for p in ps]}


def plane_start_study(p: float, instances: int, n: int = 100, starts: int = 100, seeds=C.SEEDS) -> dict:
    """Ebene, 70 % N(0, I) und 30 % N((8; 0), I), p < 1: `starts` Zufallsstarts je Instanz. Das beste Minimum ist das beste von allen Kunden und allen Enden.
    B: Anteil der Enden innerhalb 0,2 davon; A: Anteil in dessen Häufung (x > 4 oder nicht)."""
    per_B, per_A = [], []
    for seed in seeds:
        draws = Draws(cell_seed(seed, "pstart", p))
        b, a = [], []
        for pts in layout_batch(draws, "Zwei Häufungen", instances, n, share=0.7, sep=8.0):
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            ends, _ = K.irls_multi(pts, p, lo + draws.uniform((starts, 2)) * (hi - lo), maxit=C.IRLS_MAXIT_LIVE)
            end_cost = (np.linalg.norm(pts[None] - ends[:, None], axis=2) ** p).sum(axis=1)
            data_cost = (np.linalg.norm(pts[None] - pts[:, None], axis=2) ** p).sum(axis=1)
            best = pts[int(np.argmin(data_cost))] if data_cost.min() <= end_cost.min() * (1 + 1e-12) else ends[int(np.argmin(end_cost))]
            b.append(np.mean(np.linalg.norm(ends - best, axis=1) <= 0.2))
            a.append(np.mean((ends[:, 0] > 4.0) == (best[0] > 4.0)))
        per_B.append(np.mean(b))
        per_A.append(np.mean(a))
    return {"p": p, "instances_per_seed": instances, "n": n, "starts": starts, "B": mean_std(per_B), "A": mean_std(per_A)}


def run_m6a() -> dict:
    return {
        "density": [density_axis_cell(s, 200, 100) for s in (0.5, 1.0, 2.0, 3.0, 4.0)],
        "start": [axis_start_study(p, 6) for p in START_P],
        "stairs": [staircase_cell(layout, 51, 30, lo, hi, step) for layout in ("Normal", "Zwei Häufungen") for lo, hi in ((0.1, 0.9), (1.1, 3.0)) for step in (0.1, 0.02, 0.004)],
    }


def run_m6b() -> dict:
    return {
        "plane_data": [below_one_plane_cell(layout, n, 50) for layout, n in (("Normal", 5), ("Normal", 10), ("Normal", 30), ("Zwei Häufungen", 30))],
        "plane_density": [density_plane_cell(s, 100, 30) for s in (1.0, 2.0, 3.0)],
        "plane_start": [plane_start_study(p, 3) for p in (0.7, 0.5, 0.2)],
    }


# ---------------------------------------------------------------- E7: die Gewichtsiteration

def increases(trace) -> bool:
    """Steigt der Zielwert irgendwo (um mehr als 1e-12 relativ)?"""
    t = np.asarray(trace, dtype=float)
    return bool((np.diff(t) > 1e-12 * np.abs(t[:-1])).any())


def irls_instance(points: np.ndarray, p: float, maxit: int = IRLS_MAXIT) -> dict:
    """Eine Instanz, ein Exponent: Newton, Gewichtsiteration ohne und mit Dämpfung, je vom Schwerpunkt aus. Iterationen, Status und ob der Zielwert steigt."""
    start = K.centroid(points)
    ref = K.newton_lp(points, p, start)
    out = {"newton_its": ref.iterations, "newton_status": ref.status}
    for key, relax in (("plain", 1.0), ("damped", K.damping(p))):
        r = K.irls(points, p, start, relax=relax, tol=1e-10, maxit=maxit, trace=True)
        out[key] = {"its": r.iterations, "status": r.status, "rises": increases(r.trace),
                    "error": float(np.linalg.norm(r.center - ref.center))}
    return out


def irls_cell(layout: str, n: int, runs: int, seeds=C.SEEDS, maxit: int = IRLS_MAXIT) -> dict:
    rows = []
    for p in IRLS_P:
        acc = {"newton": [], "plain": [], "damped": []}
        conv = {"plain": [], "damped": []}
        rise = {"plain": [], "damped": []}
        for seed in seeds:
            res = [irls_instance(pts, p, maxit) for pts in layout_batch(Draws(cell_seed(seed, "irls", layout, n)), layout, runs, n)]
            for key in ("plain", "damped"):
                conv[key].append(np.mean([r[key]["status"] == "ok" for r in res]))
                rise[key].append(np.mean([r[key]["rises"] for r in res]))
                acc[key] += [r[key]["its"] for r in res if r[key]["status"] == "ok"]
            acc["newton"] += [r["newton_its"] for r in res]
        row = {"p": p}
        for key in ("plain", "damped"):
            its = np.array(acc[key], dtype=float)
            row[key] = {"converged": mean_std(conv[key])[0], "rises": mean_std(rise[key])[0],
                        "median_its": float(np.median(its)) if len(its) else None, "p95_its": float(np.percentile(its, 95)) if len(its) else None,
                        "max_its": float(its.max()) if len(its) else None}
        row["newton_median_its"] = float(np.median(acc["newton"]))
        rows.append(row)
    return {"layout": layout, "n": n, "runs_per_seed": runs, "maxit": maxit, "rows": rows}


def run_m7() -> dict:
    return {"cells": [irls_cell(layout, IRLS_N, 40) for layout in ("Normal", "Zwei Häufungen")]}


PARTS = {"m1a": run_m1a, "m1b": run_m1b, "m2": run_m2, "m4": run_m4, "m5": run_m5, "m6a": run_m6a, "m6b": run_m6b, "m7": run_m7}
