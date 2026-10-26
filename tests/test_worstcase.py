"""Worst-Case-Test am Slider-Maximum (DEMO-PLAYBOOK §1c): keine Zeit-Assertion, nur ein Timeout im eigenen Prozess als Netz gegen Hänger und Aufblähen.

Komplexität am Maximum n = 300: Kosten und Abstände O(n), Newton und Gewichtsiteration je Schritt O(n), p < 1: Abstandsmatrix der Kunden O(n²) und 20 Zufallsstarts O(20·n) je Schritt."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import math
import numpy as np
import lpm_constants as C
import lpm_evaluation as V

for p in (0.1, 0.9, 1.0, 2.0, 64.0, math.inf):
    for layout in ("Zwei Häufungen", "Außenposten"):
        a = V.analyse(V.Settings(layout=layout, n=C.N_MAX, p=p, solver=C.SOLVERS[2]))
        assert a.points.shape == (C.N_MAX, 2) and np.isfinite(a.regrets).all(), (p, layout)
print("fertig")
"""


def test_largest_instance_for_every_regime_finishes():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")
