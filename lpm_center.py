"""Die Lp-Mitte c(p) = argmin_c Σ ‖x_i − c‖^p (Exponent p > 0, gleiche Gewichte) und ihre Löser.

p = 2: Schwerpunkt; p = 1: geometrischer Median (Löser kopiert aus Stück 5, `lpm_geomed`); p → ∞: Mitte des kleinsten umschließenden Kreises (1-Center).
Für p > 1 ist die Zielfunktion streng konvex (eine Mitte), für p < 1 nicht (jeder Datenpunkt ist ein lokales Minimum).

Einheiten (je eine Regel, DEMO-PLAYBOOK §1b), alle ohne Zugriff auf Streamlit und ohne verstecktes Zustandsobjekt:
`lp_cost` (Kosten zum Exponenten q), `newton_lp` (gedämpftes Newton für p > 1), `irls` (Gewichtsiteration, optional gedämpft, einzelner Start),
`irls_multi` (viele Starts auf einmal), `damping` (die Dämpfung 1/(p − 1)), `solve_1d` (exakt auf der Achse), `one_center` (kleinster umschließender Kreis, Welzl),
`best_below_one` (p < 1 in der Ebene: alle Datenpunkte und Zufallsstarts), `lp_center` (wählt je nach p). Zufall nur über einen übergebenen Seed (SplitMix64-Konvention)."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from lpm_geomed import vardi_zhang, weighted_centroid
from lpm_rng import SplitMix64
from lpm_scenario import Draws

IRLS_FLOOR = 1e-12     # Boden für Abstände in den Gewichten der Gewichtsiteration (km)


class Solve(NamedTuple):
    center: np.ndarray
    iterations: int
    status: str                       # "ok", "maxit" (Grenze erreicht), "divergent" (Zahlenüberlauf), "stillstand" (Newton findet keinen Abstieg)
    trace: list[float] | None         # Zielwert Σ d^p vor jedem Schritt (nur bei trace=True)


def distances(points: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Abstände aller Kunden zum Punkt c."""
    return np.linalg.norm(points - c, axis=-1)


def lp_cost(points: np.ndarray, c: np.ndarray, q: float) -> float:
    """Kosten zum Exponenten q > 0: Σ d^q; für q = ∞ der längste Weg max d."""
    d = distances(points, c)
    if q == math.inf:
        return float(d.max())
    return float((d ** q).sum())


def centroid(points: np.ndarray) -> np.ndarray:
    """Schwerpunkt (Mitte für p = 2)."""
    return weighted_centroid(points, np.ones(len(points)))


def damping(p: float) -> float:
    """Dämpfung der Gewichtsiteration: 1/(p − 1) für p > 2, sonst 1 (keine). Eigene Herleitung (siehe 📐): der Schritt der Gewichtsiteration ist auf der Achse
    das (p − 1)-fache des Newton-Schritts."""
    return min(1.0, 1.0 / (p - 1.0)) if p > 1.0 else 1.0


def newton_lp(points: np.ndarray, p: float, c0=None, tol: float = 1e-13, maxit: int = 500) -> Solve:
    """Gedämpftes Newton-Verfahren (Armijo-Rückverfolgung) für den Minimierer von Σ ‖x − c‖^p mit endlichem p > 1.

    Rechnet in Koordinaten, die auf die Ausdehnung der Kunden normiert sind (nötig für große p). Start c0 (None: Schwerpunkt).
    Status "ok": Schritt kürzer als tol oder die Nettokraft ist gegen die Summe der Einzelkräfte verschwindend (Start schon im Optimum)."""
    if not 1.0 < p < math.inf:
        raise ValueError("newton_lp braucht einen endlichen Exponenten p > 1")
    points = np.asarray(points, dtype=float)
    m = points.mean(axis=0)
    s = float(np.linalg.norm(points - m, axis=1).max())
    if s == 0.0:                       # alle Kunden am selben Ort: die Mitte ist dieser Ort
        return Solve(m, 0, "ok", None)
    q = (points - m) / s
    dim = q.shape[1]
    c = np.zeros(dim) if c0 is None else (np.asarray(c0, dtype=float) - m) / s
    f = float((np.linalg.norm(q - c, axis=1) ** p).sum())
    for it in range(1, maxit + 1):
        diff = c - q
        d = np.maximum(np.linalg.norm(diff, axis=1), 1e-300)
        u = diff / d[:, None]
        force = p * d ** (p - 1)
        g = (force[:, None] * u).sum(axis=0)
        if np.linalg.norm(g) <= 1e-13 * force.sum():
            return Solve(c * s + m, it - 1, "ok", None)
        a = p * d ** (p - 2)
        hess = a.sum() * np.eye(dim) + (u.T * (a * (p - 2))) @ u
        step = -np.linalg.solve(hess + 1e-14 * np.trace(hess) * np.eye(dim), g)
        slope = float(g @ step)
        if slope >= 0:
            step, slope = -g, -float(g @ g)
        t = 1.0
        while t > 1e-16:
            f_new = float((np.linalg.norm(q - (c + t * step), axis=1) ** p).sum())
            if f_new <= f + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            return Solve(c * s + m, it, "stillstand", None)
        c_new = c + t * step
        done = float(np.linalg.norm(c_new - c)) < tol
        c, f = c_new, f_new
        if done:
            return Solve(c * s + m, it, "ok", None)
    return Solve(c * s + m, maxit, "maxit", None)


def irls(points: np.ndarray, p: float, c0, relax: float = 1.0, tol: float = 1e-10, maxit: int = 20000, trace: bool = False) -> Solve:
    """Gewichtsiteration c ← Σ w_i x_i / Σ w_i mit w_i = d_i^(p−2) (d_i ≥ IRLS_FLOOR); relax < 1 dämpft: c ← c + relax·(c_neu − c).

    p = 1 ist das Weiszfeld-Verfahren, p = 2 der Schwerpunkt in einem Schritt. Abbruch, wenn der Schritt kürzer als tol ist. Status "maxit" ist ein Ergebnis, kein Fehler:
    ohne Dämpfung konvergiert das Verfahren für größere p (gemessen ab p = 4) nicht. `trace`: der Zielwert Σ d^p vor jedem Schritt."""
    points = np.asarray(points, dtype=float)
    c = np.array(c0, dtype=float)
    values: list[float] | None = [] if trace else None
    for it in range(1, maxit + 1):
        d = np.maximum(np.linalg.norm(points - c, axis=1), IRLS_FLOOR)
        if values is not None:
            values.append(float((d ** p).sum()))
        w = (d / d.max()) ** (p - 2.0)
        c_new = c + relax * ((points * w[:, None]).sum(axis=0) / w.sum() - c)
        if not np.all(np.isfinite(c_new)):
            return Solve(c, it, "divergent", values)
        step = float(np.linalg.norm(c_new - c))
        c = c_new
        if step < tol:
            return Solve(c, it, "ok", values)
    return Solve(c, maxit, "maxit", values)


def irls_multi(points: np.ndarray, p: float, starts: np.ndarray, tol: float = 1e-9, maxit: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    """Gewichtsiteration (ungedämpft) für viele Starts gleichzeitig: starts (S, dim) → (Enden (S, dim), Schrittzahlen (S,)). Wer die Toleranz erreicht, scheidet aus."""
    ends = np.array(starts, dtype=float)
    count = np.zeros(len(ends), dtype=int)
    active = np.ones(len(ends), dtype=bool)
    for it in range(1, maxit + 1):
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            break
        d = np.maximum(np.linalg.norm(points[None, :, :] - ends[idx][:, None, :], axis=2), IRLS_FLOOR)
        w = (d / d.max(axis=1, keepdims=True)) ** (p - 2.0)
        new = np.einsum("sn,nd->sd", w, points) / w.sum(axis=1)[:, None]
        step = np.linalg.norm(new - ends[idx], axis=1)
        ends[idx] = new
        count[idx] = it
        active[idx[step < tol]] = False
    return ends, count


def solve_1d(x, p: float) -> tuple[float, float]:
    """Exakte Mitte auf der Achse als Intervall (untere, obere Grenze).

    p = ∞: Mitte der Spannweite. p = 1: Intervall der Mediane (bei gerader Anzahl [x_(n/2), x_(n/2+1)], sonst ein Punkt). 1 < p < ∞: die Nullstelle der strikt wachsenden
    Ableitung Σ sign(c − x)|c − x|^(p−1), per Bisektion (ein Punkt). 0 < p < 1: ein Datenpunkt, denn zwischen zwei Datenpunkten ist Σ|c − x|^p konkav;
    gewählt wird der mit den kleinsten Kosten, bei Gleichstand (Kosten gleich bis 1e-12 relativ) der kleinere Wert."""
    if p <= 0:
        raise ValueError("p muss größer als 0 sein")
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if p == math.inf:
        mid = float((x[0] + x[-1]) / 2)
        return mid, mid
    if p == 1.0:
        return float(x[(n - 1) // 2]), float(x[n // 2])
    if p < 1.0:
        costs = (np.abs(x[:, None] - x[None, :]) ** p).sum(axis=0)
        best = int(np.flatnonzero(costs <= costs.min() * (1.0 + 1e-12))[0])
        return float(x[best]), float(x[best])
    mid, half = (x[0] + x[-1]) / 2.0, (x[-1] - x[0]) / 2.0
    if half == 0.0:
        return float(x[0]), float(x[0])
    y = (x - mid) / half
    lo, hi = -1.0, 1.0
    for _ in range(100):
        c = (lo + hi) / 2.0
        if (np.sign(c - y) * np.abs(c - y) ** (p - 1.0)).sum() > 0:
            hi = c
        else:
            lo = c
    c = mid + half * (lo + hi) / 2.0
    return float(c), float(c)


def _circle_two(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    return (a + b) / 2.0, float(np.linalg.norm(a - b) / 2.0)


def _circle_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    """Umkreis dreier Punkte; sind sie kollinear (Nenner 0), der größte Kreis über zwei von ihnen."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0.0:
        return max((_circle_two(u, v) for u, v in ((a, b), (a, c), (b, c))), key=lambda t: t[1])
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    center = np.array([ux, uy])
    return center, float(np.linalg.norm(a - center))


def one_center(points: np.ndarray, seed: int = 0) -> tuple[np.ndarray, float]:
    """Kleinster umschließender Kreis (1-Center, Mitte für p → ∞): Mittelpunkt und Radius. Inkrementelle Fassung nach Welzl (drei verschachtelte Schleifen);
    die Reihenfolge der Punkte ist eine mit `seed` gemischte Permutation (SplitMix64), das Ergebnis hängt davon nicht ab (der Kreis ist eindeutig)."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    order = list(range(n))
    rng = SplitMix64(seed)
    for i in range(n - 1, 0, -1):
        j = rng.below(i + 1)
        order[i], order[j] = order[j], order[i]
    q = pts[order]
    eps = 1e-12 * float(np.ptp(pts, axis=0).max())
    center, radius = q[0].copy(), 0.0
    for i in range(1, n):
        if np.linalg.norm(q[i] - center) <= radius + eps:
            continue
        center, radius = q[i].copy(), 0.0
        for j in range(i):
            if np.linalg.norm(q[j] - center) <= radius + eps:
                continue
            center, radius = _circle_two(q[i], q[j])
            for k in range(j):
                if np.linalg.norm(q[k] - center) > radius + eps:
                    center, radius = _circle_three(q[i], q[j], q[k])
    return center, radius


class BelowOne(NamedTuple):
    center: np.ndarray
    cost: float
    on_data: bool                     # das beste gefundene Minimum liegt auf einem Kunden
    ends: np.ndarray                  # Enden der Zufallsstarts (S, 2)
    data_cost: float                  # Kosten des besten Kunden


def best_below_one(points: np.ndarray, p: float, seed: int = 0, starts: int = 20, maxit: int = 3000) -> BelowOne:
    """Mitte für 0 < p < 1 in der Ebene: der beste Kunde, verglichen mit den Enden der Gewichtsiteration aus `starts` Zufallsstarts im Rahmen der Kunden.

    Die Kosten haben in jedem Kunden eine Spitze (jeder Kunde ist ein lokales Minimum); im Innern kann es ein weiteres Minimum geben (gemessen für p nahe 1).
    Es wird nur das beste von endlich vielen Kandidaten gefunden, kein Beweis des globalen Minimums. Bei Gleichstand gewinnt der Kunde."""
    if not 0.0 < p < 1.0:
        raise ValueError("best_below_one gilt für 0 < p < 1")
    pts = np.asarray(points, dtype=float)
    pair = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    data_costs = (pair ** p).sum(axis=1)
    k = int(np.argmin(data_costs))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ends, _ = irls_multi(pts, p, lo + Draws(seed).uniform((starts, 2)) * (hi - lo), maxit=maxit)
    end_costs = (np.linalg.norm(pts[None, :, :] - ends[:, None, :], axis=2) ** p).sum(axis=1)
    e = int(np.argmin(end_costs))
    if end_costs[e] < data_costs[k] * (1.0 - 1e-12):
        return BelowOne(ends[e].copy(), float(end_costs[e]), False, ends, float(data_costs[k]))
    return BelowOne(pts[k].copy(), float(data_costs[k]), True, ends, float(data_costs[k]))


def lp_center(points: np.ndarray, p: float, seed: int = 0) -> np.ndarray:
    """Mitte zum Exponenten p in der Ebene: ∞ → 1-Center, 1 → Vardi–Zhang, 2 → Schwerpunkt, p > 1 sonst Newton, p < 1 → bester Kunde bzw. Zufallsstart-Ende."""
    pts = np.asarray(points, dtype=float)
    if p == math.inf:
        return one_center(pts, seed)[0]
    if p == 1.0:
        return vardi_zhang(pts, np.ones(len(pts)), tol=1e-12).center
    if p == 2.0:
        return centroid(pts)
    if p > 1.0:
        return newton_lp(pts, p).center
    return best_below_one(pts, p, seed).center
