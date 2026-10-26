"""Schleifen mit Abbruchbedingung (Armijo-Rückverfolgung, Iterationen, Bisektion, Welzl) enden im eigenen Prozess innerhalb einer Zeitgrenze.

Ein Mutant, der eine solche Schleife endlos laufen lässt, wird sonst erst vom Zeitlimit des Mutationswerkzeugs gefangen (und nicht in tests/mutation_state.json vermerkt)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = r"""
import math
import numpy as np
import lpm_center as K
import lpm_evaluation as V
import lpm_experiments as X
from lpm_scenario import make_points

for layout in ("Normal", "Außenposten", "Zwei Häufungen", "Quadrat"):
    pts = make_points(layout, 40, 3)
    for p in (1.2, 2.0, 3.0, 8.0, 40.0, 128.0):
        if p != 2.0:
            assert K.newton_lp(pts, p).status in ("ok", "stillstand", "maxit")
    assert K.irls(pts, 6.0, K.centroid(pts), maxit=300).iterations <= 300
    assert K.irls(pts, 6.0, K.centroid(pts), relax=0.2).status == "ok"
    for p in (0.3, 0.7, 0.95):
        K.best_below_one(pts, p, 1, starts=5, maxit=300)
    K.one_center(pts, 5)
    K.lp_center(pts, 1.0)
x = np.random.default_rng(1).normal(size=51)
for p in (0.2, 1.0, 1.01, 3.0, 64.0, math.inf):
    K.solve_1d(x, p)
for p in (0.5, 1.0, 1.5, 4.0, 16.0, math.inf):
    V.analyse(V.Settings(p=p, n=60))
X.irls_instance(make_points("Normal", 30, 2), 5.0, 400)
print("fertig")
"""


def test_all_iterations_terminate_within_the_time_limit():
    done = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=120, encoding="utf-8")
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().endswith("fertig")
