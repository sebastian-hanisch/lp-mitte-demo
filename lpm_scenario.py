"""Szenario: Kunden in der Ebene (1 Einheit = 1 km, gleiche Gewichte) und auf der Zahlenachse, erzeugt aus dem eigenen SplitMix64-Strom (lpm_rng.py).

`Draws` zieht Zahlen blockweise (numpy, uint64 rechnet modulo 2^64) und liefert exakt den Strom der wörtlich kopierten Klasse `SplitMix64`;
tests/test_scenario.py vergleicht beide. Damit sind alle Instanzen auf jeder Plattform gleich, und Messreihen über viele Instanzen bleiben schnell.

Layouts der Ebene: Normal (N(0, I)), Quadrat (gleichverteilt auf [-1, 1]²), Außenposten (Kunde 0 steht bei (a, a), die übrigen n - 1 sind normalverteilt),
Zwei Häufungen (der Anteil `share` der Kunden, die ersten, ist N(0, spread²·I), der Rest N((sep; 0), I)).
Layouts der Achse: Normal, Lognormal, Außenposten (n - 1 Normalwerte plus ein Wert bei `outpost`), Zwei Häufungen (Anteil `share` um 0, Rest um `sep`, beide Streuung 1),
Dichte (der Anteil `share` um 0 mit Streuung `s_big`, der Rest um `sep` mit Streuung `s_small`)."""

from __future__ import annotations

import math
import zlib

import numpy as np

from lpm_rng import MASK

GAMMA = 0x9E3779B97F4A7C15


class Draws:
    """SplitMix64-Strom, blockweise gezogen: gleicher Seed, gleiche Zahlen wie `lpm_rng.SplitMix64`, aber ohne Python-Schleife."""

    def __init__(self, seed: int):
        self.state = int(seed) & MASK

    def raw(self, count: int) -> np.ndarray:
        """Die nächsten `count` Ausgaben des Stroms als uint64-Feld."""
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.arange(1, count + 1, dtype=np.uint64) * np.uint64(GAMMA) + np.uint64(self.state)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + count * GAMMA) & MASK
        return z

    def uniform(self, shape) -> np.ndarray:
        """Gleichverteilt auf [0, 1) mit 53 Bit."""
        size = int(np.prod(shape))
        return ((self.raw(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))).reshape(shape)

    def normal(self, shape) -> np.ndarray:
        """Standardnormal nach Box-Muller: zwei gleichverteilte Zahlen ergeben ein Paar (Kosinus-Wert zuerst)."""
        size = int(np.prod(shape))
        pairs = (size + 1) // 2
        u = self.uniform((2, pairs))
        radius = np.sqrt(-2.0 * np.log(1.0 - u[0]))        # 1 - u liegt in (0, 1]: kein log(0)
        angle = 2.0 * math.pi * u[1]
        z = np.concatenate([radius * np.cos(angle), radius * np.sin(angle)])
        return z[:size].reshape(shape)


def cell_seed(base: int, *tags) -> int:
    """Stabiler Seed einer Zelle aus dem Basis-Seed und beliebig vielen Marken (Zahlen oder Texte)."""
    value = int(base)
    for tag in tags:
        part = zlib.crc32(tag.encode("utf-8")) if isinstance(tag, str) else int(round(float(tag) * 1000))
        value = (value * 1_000_003 + part) % (1 << 61)
    return value


def group_size(share: float, n: int) -> int:
    """Größe der großen Häufung: Anteil mal n, kaufmännisch gerundet, mindestens 1 und höchstens n - 1 (die kleine Häufung ist nie leer)."""
    return int(min(max(int(share * n + 0.5), 1), n - 1))


def layout_batch(draws: Draws, layout: str, runs: int, n: int, outpost: float = 6.0, share: float = 0.7, sep: float = 6.0, spread: float = 1.0) -> np.ndarray:
    """`runs` Punktmengen mit je n Kunden, Form (runs, n, 2)."""
    if layout == "Normal":
        return draws.normal((runs, n, 2))
    if layout == "Quadrat":
        return draws.uniform((runs, n, 2)) * 2.0 - 1.0
    if layout == "Außenposten":
        points = draws.normal((runs, n, 2))
        points[:, 0] = (outpost, outpost)
        return points
    if layout == "Zwei Häufungen":
        points = draws.normal((runs, n, 2))
        k = group_size(share, n)
        points[:, :k] *= spread
        points[:, k:, 0] += sep
        return points
    raise ValueError(f"unbekanntes Layout: {layout}")


def make_points(layout: str, n: int, seed: int, outpost: float = 6.0, share: float = 0.7, sep: float = 6.0, spread: float = 1.0) -> np.ndarray:
    """Eine Instanz (n, 2), deterministisch aus Layout, n, Parametern und Zufalls-Seed."""
    return layout_batch(Draws(seed), layout, 1, n, outpost, share, sep, spread)[0]


AXIS_LAYOUTS = ("Normal", "Lognormal", "Außenposten", "Zwei Häufungen")


def axis_batch(draws: Draws, layout: str, runs: int, n: int, outpost: float = 6.0, share: float = 0.7, sep: float = 6.0) -> np.ndarray:
    """`runs` Mengen von n Werten auf der Achse, Form (runs, n)."""
    z = draws.normal((runs, n))
    if layout == "Normal":
        return z
    if layout == "Lognormal":
        return np.exp(z)
    if layout == "Außenposten":
        z[:, 0] = outpost
        return z
    if layout == "Zwei Häufungen":
        z[:, group_size(share, n):] += sep
        return z
    raise ValueError(f"unbekanntes Achsen-Layout: {layout}")


def density_batch(draws: Draws, runs: int, n: int, s_big: float, share: float = 0.6, sep: float = 8.0, s_small: float = 0.5) -> np.ndarray:
    """Die große Häufung (Anteil `share`) um 0 mit Streuung s_big, die kleine um `sep` mit Streuung s_small; Form (runs, n)."""
    z = draws.normal((runs, n))
    k = group_size(share, n)
    z[:, :k] *= s_big
    z[:, k:] = z[:, k:] * s_small + sep
    return z
