"""Orakel-Tests der Lp-Mitte gegen andere Rechenwege (ORACLE-PLAYBOOK §2): scipy.optimize (mit analytischem Gradienten), Gitter, Aufzählung aller Paare und Tripel,
Nullstelle der Ableitung per Brent-Verfahren, geschlossene Achsenformel (tests/test_center.py).

Der große Lauf (Scratch, nicht im Repo, 2026-10-25): Newton gegen scipy je 500 Instanzen für p = 1,2 bis 30, 1-Center gegen Aufzählung an 3 000 Instanzen,
Achse gegen ein Gitter mit 20 001 Punkten; Ergebnisse stehen in der README unter „Verifikation“. Hier die kleine, schnelle Fassung."""

import math
from itertools import combinations

import numpy as np
import pytest

import lpm_center as K
import lpm_regret as RG
from lpm_scenario import Draws, density_batch, layout_batch

sp = pytest.importorskip("scipy.optimize")


def _instances(count, n_lo, n_hi, seed):
    draws = Draws(seed)
    out = []
    for i in range(count):
        n = n_lo + i % (n_hi - n_lo + 1)
        layout = ("Normal", "Außenposten", "Zwei Häufungen", "Quadrat")[i % 4]
        out.append(layout_batch(draws, layout, 1, n)[0])
    return out


def _scipy_minimum(points, p):
    """Minimum von Σ d^p mit BFGS und analytischem Gradienten in auf die Ausdehnung normierten Koordinaten (Referenz mit Startpunkt Schwerpunkt)."""
    m = points.mean(axis=0)
    s = np.linalg.norm(points - m, axis=1).max()
    q = (points - m) / s

    def f(c):
        return float((np.linalg.norm(q - c, axis=1) ** p).sum())

    def grad(c):
        diff = c - q
        d = np.maximum(np.linalg.norm(diff, axis=1), 1e-300)
        return (p * (d ** (p - 2))[:, None] * diff).sum(axis=0)

    res = sp.minimize(f, np.zeros(2), jac=grad, method="BFGS", options={"gtol": 1e-12, "maxiter": 2000})
    return res.fun * s ** p, res.x * s + m


@pytest.mark.parametrize("p", [1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])
def test_newton_is_never_worse_than_scipy_and_satisfies_the_optimality_condition(p):
    for points in _instances(16, 3, 30, seed=int(p * 10)):
        ours = K.newton_lp(points, p).center
        f_ours = float((np.linalg.norm(points - ours, axis=1) ** p).sum())
        f_ref, _ = _scipy_minimum(points, p)
        assert f_ours <= f_ref * (1 + 1e-9)
        # unabhängige Optimalitätsbedingung: die Kräfte p·d^(p−1)·Einheitsvektor heben sich auf (relativ zur Summe der Einzelkräfte)
        diff = ours - points
        d = np.linalg.norm(diff, axis=1)
        force = (p * d ** (p - 2))[:, None] * diff
        assert np.linalg.norm(force.sum(axis=0)) <= 1e-8 * (p * d ** (p - 1)).sum()


def test_solve_1d_above_one_is_the_root_of_the_derivative():
    draws = Draws(5)
    for x in draws.normal((30, 15)):
        for p in (1.1, 1.5, 2.0, 3.0, 7.0):
            lo, hi = K.solve_1d(x, p)

            def deriv(c, p=p, x=x):
                return float((np.sign(c - x) * np.abs(c - x) ** (p - 1)).sum())

            root = sp.brentq(deriv, x.min() - 1e-9, x.max() + 1e-9, xtol=1e-14, rtol=1e-14)
            assert lo == hi == pytest.approx(root, abs=1e-9)


def test_solve_1d_is_never_worse_than_a_fine_grid_for_any_exponent():
    draws = Draws(8)
    for x in draws.normal((30, 11)):
        grid = np.linspace(x.min(), x.max(), 20001)
        for p in (0.2, 0.5, 0.9, 1.5, 3.0, 8.0):
            lo, hi = K.solve_1d(x, p)
            cost = lambda c: (np.abs(c[..., None] - x) ** p).sum(axis=-1)
            assert cost(np.array(lo)) <= cost(grid).min() + 1e-12 * max(1.0, cost(grid).min())
    # Achse mit Außenposten und Lognormal ebenso
    for x in density_batch(Draws(9), 20, 21, s_big=2.0):
        grid = np.linspace(x.min(), x.max(), 20001)
        for p in (0.3, 0.7):
            lo, _ = K.solve_1d(x, p)
            assert (np.abs(lo - x) ** p).sum() <= (np.abs(grid[:, None] - x) ** p).sum(axis=1).min() + 1e-9


def test_one_center_equals_the_smallest_circle_over_all_pairs_and_triples():
    for points in _instances(80, 3, 10, seed=21):
        center, radius = K.one_center(points)
        best = math.inf
        for i, j in combinations(range(len(points)), 2):
            c, r = (points[i] + points[j]) / 2, np.linalg.norm(points[i] - points[j]) / 2
            if np.linalg.norm(points - c, axis=1).max() <= r + 1e-9:
                best = min(best, r)
        for i, j, k in combinations(range(len(points)), 3):
            a, b, cc = points[i], points[j], points[k]
            d = 2 * (a[0] * (b[1] - cc[1]) + b[0] * (cc[1] - a[1]) + cc[0] * (a[1] - b[1]))
            if abs(d) < 1e-12:
                continue
            ux = ((a @ a) * (b[1] - cc[1]) + (b @ b) * (cc[1] - a[1]) + (cc @ cc) * (a[1] - b[1])) / d
            uy = ((a @ a) * (cc[0] - b[0]) + (b @ b) * (a[0] - cc[0]) + (cc @ cc) * (b[0] - a[0])) / d
            c = np.array([ux, uy])
            r = np.linalg.norm(a - c)
            if np.linalg.norm(points - c, axis=1).max() <= r + 1e-9:
                best = min(best, r)
        assert radius == pytest.approx(best, abs=1e-9)
        assert np.linalg.norm(points - center, axis=1).max() == pytest.approx(radius, abs=1e-9)


def test_one_center_radius_is_never_beaten_by_scipy():
    for points in _instances(8, 20, 40, seed=33):
        _, radius = K.one_center(points)
        for start in (points.mean(axis=0), points[0], points[-1]):
            res = sp.minimize(lambda c: np.linalg.norm(points - c, axis=1).max(), start, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
            assert res.fun >= radius - 1e-9


def test_gewichtsiteration_agrees_with_newton_where_it_converges():
    for points in _instances(12, 4, 30, seed=44):
        start = points.mean(axis=0)
        for p, relax in ((1.2, 1.0), (1.5, 1.0), (2.5, 1.0), (3.0, K.damping(3.0)), (4.0, K.damping(4.0)), (6.0, K.damping(6.0))):
            r = K.irls(points, p, start, relax=relax, tol=1e-12, maxit=20000)
            assert r.status == "ok", (p, relax)
            assert r.center == pytest.approx(K.newton_lp(points, p).center, abs=1e-6)


def test_best_below_one_is_never_worse_than_a_fine_grid():
    gx = np.linspace(-1.0, 1.0, 161)
    for i, points in enumerate(_instances(24, 3, 5, seed=55)):
        points = points / max(1.0, np.abs(points).max())          # Gitter über [−1, 1]²
        gridxy = np.stack(np.meshgrid(gx, gx), axis=-1).reshape(-1, 2)
        for p in (0.5, 0.8, 0.95):
            r = K.best_below_one(points, p, seed=i)
            grid_cost = (np.linalg.norm(gridxy[:, None, :] - points[None], axis=2) ** p).sum(axis=1).min()
            assert r.cost <= grid_cost * (1 + 1e-9)


def test_regret_matrix_against_scipy_minimisers():
    for points in _instances(10, 5, 25, seed=66):
        mat = RG.regret_matrix(points)
        c1 = sp.minimize(lambda c: np.linalg.norm(points - c, axis=1).sum(), points.mean(axis=0), method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 5000}).x
        c2 = points.mean(axis=0)
        cinf = sp.minimize(lambda c: np.linalg.norm(points - c, axis=1).max(), points.mean(axis=0), method="Nelder-Mead", options={"xatol": 1e-11, "fatol": 1e-13, "maxiter": 5000}).x
        centers = (c1, c2, cinf)
        cost = lambda c, q: np.linalg.norm(points - c, axis=1).sum() if q == 1 else ((np.linalg.norm(points - c, axis=1) ** 2).sum() if q == 2 else np.linalg.norm(points - c, axis=1).max())
        for a, ca in enumerate(centers):
            for b, q in enumerate((1, 2, 3)):
                expected = 100 * (cost(ca, q) / cost(centers[b], q) - 1)
                assert mat[a, b] == pytest.approx(expected, abs=2e-3)
        assert np.diag(mat) == pytest.approx([0, 0, 0], abs=1e-9)
