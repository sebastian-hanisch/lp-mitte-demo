"""Reue (Regret) einer Wahl: Wer die Mitte nach dem Exponenten p wählt und nach dem Kriterium q bewertet, zahlt gegenüber dem Optimum für q Mehrkosten.

Regret_{p→q} = Kosten_q(c_p) / Kosten_q(c_q) − 1 (hier in Prozent). Kriterien: q = 1 Summe der Wege, q = 2 Quadratsumme, q = ∞ längster Weg;
c_1 geometrischer Median, c_2 Schwerpunkt, c_∞ exaktes 1-Center. Einheiten: `optimal_centers`, `regret_row`, `regret_matrix`, `compromise_regrets`, `empty_zone`."""

from __future__ import annotations

import math

import numpy as np

from lpm_center import lp_center, lp_cost

Q_VALUES = (1.0, 2.0, math.inf)


def optimal_centers(points: np.ndarray, seed: int = 0) -> dict[float, np.ndarray]:
    """Die Mitten, die die drei Kriterien minimieren: {1: geometrischer Median, 2: Schwerpunkt, ∞: 1-Center}."""
    return {q: lp_center(points, q, seed) for q in Q_VALUES}


def regret_percent(cost: float, optimum: float) -> float:
    """(Kosten / Optimum − 1) in Prozent; sind die optimalen Kosten 0 (alle Kunden am selben Ort), gibt es keine Mehrkosten."""
    return 0.0 if optimum <= 0.0 else 100.0 * (cost / optimum - 1.0)


def regret_row(points: np.ndarray, c: np.ndarray, optimum: dict[float, np.ndarray]) -> np.ndarray:
    """Regret der Mitte c bewertet nach q = 1, 2, ∞ (drei Zahlen in Prozent)."""
    return np.array([regret_percent(lp_cost(points, c, q), lp_cost(points, optimum[q], q)) for q in Q_VALUES])


def regret_matrix(points: np.ndarray, seed: int = 0) -> np.ndarray:
    """3×3-Matrix [Zeile p = 1, 2, ∞][Spalte q = 1, 2, ∞] der Regrets in Prozent (Diagonale 0)."""
    optimum = optimal_centers(points, seed)
    return np.array([regret_row(points, optimum[p], optimum) for p in Q_VALUES])


def compromise_regrets(points: np.ndarray, grid, seed: int = 0) -> np.ndarray:
    """Für jedes p des Rasters der größte Regret über die drei Kriterien (Prozent): die Kenngröße des Kompromisses."""
    optimum = optimal_centers(points, seed)
    out = []
    for p in grid:
        c = optimum[p] if p in optimum else lp_center(points, p, seed)
        out.append(float(regret_row(points, c, optimum).max()))
    return np.array(out)


def empty_zone(points: np.ndarray, c: np.ndarray, radius: float) -> bool:
    """Liegt die Mitte in einer leeren Zone, d. h. ist der nächste Kunde weiter als `radius` entfernt?"""
    return bool(np.linalg.norm(points - c, axis=1).min() > radius)
