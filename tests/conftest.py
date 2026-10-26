"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Die Erwartungswerte stehen im Test als Rechnung.

Handrechnungen (gleiche Gewichte):
* Achse 0, 0, 0, 12: f(c) = 3c^p + (12 − c)^p, f'(c) = 0 ⇔ 3c^(p−1) = (12 − c)^(p−1) ⇔ c(p) = 12 / (1 + 3^(1/(p−1))) für jedes p > 1.
  p = 1,5: 3² = 9, c = 12/10 = 1,2. p = 2: c = 12/4 = 3 (Mittelwert). p = 3: c = 12/(1 + √3) = 4,3923. p → ∞: 3⁰ = 1, c = 6 (Mitte der Spannweite).
* Achse 1, 2, 3, 10: p = 2: 16/4 = 4. p = 3: (c−1)² + (c−2)² + (c−3)² = (10−c)² ⇔ 2c² + 8c − 86 = 0 ⇔ c = −2 + √47 = 4,8557. p = ∞: 5,5. p = 1: Intervall [2; 3].
* Achse 0, 1, 10, 11, p = 0,5: Kosten in 1: 1 + 0 + √9 + √10 = 7,1623, in 10 genauso (Spiegelbild), in 0 und 11: 1 + √10 + √11 = 7,4789. Zwei gleich gute Minima in 1 und 10.
* Ebene (0, 0), (4, 0), (0, 3): rechtwinkliges Dreieck, Hypotenuse 5, 1-Center (2; 1,5) mit Radius 2,5 (alle drei Ecken im Abstand 2,5). Schwerpunkt (4/3; 1).
  Summe der Wege im 1-Center 7,5, im Schwerpunkt 5/3 + √73/3 + √52/3 = 6,9184; Quadratsumme 18,75 gegen 150/9 = 16,667; längster Weg im Schwerpunkt √73/3 = 2,8480.
* zwei Punkte: für jedes p > 1 ist die Mitte der Mittelpunkt der Strecke (Symmetrie).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def axis_a():
    """Achse 0, 0, 0, 12 als (4, 1)-Feld."""
    return np.array([[0.0], [0.0], [0.0], [12.0]])


@pytest.fixture
def axis_b():
    return np.array([1.0, 2.0, 3.0, 10.0])


@pytest.fixture
def axis_ties():
    return np.array([0.0, 1.0, 10.0, 11.0])


@pytest.fixture
def right_triangle():
    return np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])


@pytest.fixture
def two_points():
    return np.array([[1.0, 2.0], [5.0, -2.0]])


def axis_a_center(p):
    """Geschlossene Formel für 0, 0, 0, 12."""
    return 12.0 / (1.0 + 3.0 ** (1.0 / (p - 1.0)))


SQRT47 = math.sqrt(47.0)
