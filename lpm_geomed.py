# Wörtliche Kopie einzelner Bausteine von weiszfeld-demo/wsz_geomed.py (COINCIDENT, Result, weighted_centroid, _start_point, vardi_zhang; letzter Commit der Datei: 85d4554,
# Arbeitsbaum sauber am 2026-10-25); keine Änderung außer diesem Kopf. Änderungen gehören ins Original, nicht hierher (tests/test_copies.py vergleicht Baustein für Baustein).
"""Der Löser für p = 1 (Vardi–Zhang, geometrischer Median) kommt als Kopie aus Stück 5 der Reihe (weiszfeld-demo); hier ist p = 1 nur eine Stelle auf dem Pfad."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

COINCIDENT = 1e-13


class Result(NamedTuple):
    center: np.ndarray
    iterations: int
    status: str
    path: np.ndarray | None


def weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Gewichteter Schwerpunkt (Minimum der Summe der quadrierten Abstände)."""
    return (points * weights[:, None]).sum(axis=0) / weights.sum()


def _start_point(points, weights, start) -> np.ndarray:
    return weighted_centroid(points, weights) if start is None else np.array(start, dtype=float)


def vardi_zhang(points, weights, start=None, tol: float = 1e-9, maxit: int = 100000, keep: int = 0) -> Result:
    """Weiszfeld-Schritt mit Behandlung des Kunden, auf dem der Iterat steht (Fassung des Plans; Originalarbeit nicht eingesehen, gegen scipy geprüft).

    Steht der Iterat auf Kunden mit Gesamtgewicht η und ist die Norm r der Zugkraft der übrigen höchstens η, ist er das Optimum (Status "datenpunkt").
    Sonst: c ← (1 − γ) T(c) + γ c mit T dem Weiszfeld-Schritt ohne die Kunden am Ort und γ = min(1, η / r)."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=float)
    c = _start_point(points, weights, start)
    path = [c.copy()] if keep > 0 else None
    if np.all(points == points[0]):
        return Result(points[0].copy(), 0, "ok", np.array(path) if path is not None else None)
    status, it = "maxit", 0
    for it in range(1, maxit + 1):
        d = np.linalg.norm(points - c, axis=1)
        on = d < COINCIDENT
        if on.any():
            eta = float(weights[on].sum())
            wi = weights[~on] / d[~on]
            t = (points[~on] * wi[:, None]).sum(axis=0) / wi.sum()
            r = float(np.linalg.norm(((points[~on] - c) * wi[:, None]).sum(axis=0)))
            if r <= eta:
                c = points[on][0].copy()
                status = "datenpunkt"
                break
            g = min(1.0, eta / r)
            c_new = (1.0 - g) * t + g * c
        else:
            wi = weights / d
            c_new = (points * wi[:, None]).sum(axis=0) / wi.sum()
        if path is not None and len(path) < keep:
            path.append(c_new.copy())
        step = float(np.linalg.norm(c_new - c))
        c = c_new
        if step < tol:
            status = "ok"
            break
    return Result(c, it, status, np.array(path) if path is not None else None)
