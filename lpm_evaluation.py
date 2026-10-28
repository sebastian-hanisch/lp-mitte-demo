"""Kennzahlen der Karte (eine Instanz, ein Exponent p) und Verdikte. Verdikte sind reine Funktionen (Art, Text); die App ruft st.<Art>(Text).

Hier steht der Buchstabe p immer für den Exponenten der Strecke (Σ d^p); das Bewertungskriterium heißt q."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

import lpm_center as K
import lpm_constants as C
import lpm_regret as RG
from lpm_scenario import Draws, cell_seed, make_points


def de(x: float, nd: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x: float, nd: int = 1) -> str:
    """Anteil (0 bis 1) als Prozentzahl mit Dezimalkomma, ohne Prozentzeichen."""
    return de(100.0 * x, nd)


def fmt_p(p: float) -> str:
    """Exponent als Text: ∞ oder die Zahl mit Dezimalkomma ohne überflüssige Nullen (0,5; 1; 1,25; 64)."""
    if p == math.inf:
        return "∞"
    return f"{p:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    layout: str = C._BASE["layout"]
    n: int = C._BASE["n"]
    outpost: float = C._BASE["outpost"]
    share: int = C._BASE["share"]
    sep: float = C._BASE["sep"]
    spread: float = C._BASE["spread"]
    p: float = C._BASE["p"]
    criterion: str = C._BASE["criterion"]
    solver: str = C._BASE["solver"]
    start: str = C._BASE["start"]
    seed: int = C._BASE["seed"]


assert [f.name for f in fields(Settings)] == list(C._BASE), "Settings und Presets müssen dieselben Regler in derselben Reihenfolge führen"


def criterion_q(label: str) -> float:
    """Exponent q des gewählten Bewertungskriteriums."""
    return C.CRITERION_Q[C.CRITERIA.index(label)]


def build_points(settings: Settings) -> np.ndarray:
    return make_points(settings.layout, settings.n, settings.seed, settings.outpost, settings.share / 100.0, settings.sep, settings.spread)


def start_point(settings: Settings, points: np.ndarray) -> np.ndarray:
    """Startpunkt der Löser: der Schwerpunkt oder ein Zufallspunkt im Rahmen der Kunden."""
    if settings.start == "Schwerpunkt":
        return K.centroid(points)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return lo + Draws(cell_seed(settings.seed, "start")).uniform(2) * (hi - lo)


def run_solver(settings: Settings, points: np.ndarray) -> K.Solve:
    """Der Lauf des gewählten Lösers für den Exponenten p der Einstellungen.

    ∞: kleinster umschließender Kreis (Welzl). 1: Vardi–Zhang. 2: Schwerpunkt. 1 < p < ∞ sonst: Newton oder Gewichtsiteration (ohne/mit Dämpfung 1/(p − 1)) vom Start aus.
    p < 1: Gewichtsiteration vom Start aus (endet auf einem Kunden oder in einem inneren lokalen Minimum, nicht unbedingt im besten)."""
    p = settings.p
    start = start_point(settings, points)
    if p == math.inf:
        return K.Solve(K.one_center(points)[0], 1, "ok", None)
    if p == 1.0:
        r = K.vardi_zhang(points, np.ones(len(points)), start=start, tol=1e-12, maxit=C.IRLS_MAXIT_LIVE * 10)
        return K.Solve(r.center, r.iterations, r.status, None)
    if p == 2.0:
        return K.Solve(K.centroid(points), 0, "ok", None)
    if p < 1.0 or settings.solver == C.SOLVERS[1]:
        return K.irls(points, p, start, relax=1.0, maxit=C.IRLS_MAXIT_LIVE)
    if settings.solver == C.SOLVERS[2]:
        return K.irls(points, p, start, relax=K.damping(p), maxit=C.IRLS_MAXIT_LIVE)
    return K.newton_lp(points, p, c0=start)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    points: np.ndarray
    centroid: np.ndarray
    median: np.ndarray                 # geometrischer Median (p = 1)
    one_center: np.ndarray
    one_radius: float
    solve: K.Solve                     # Lauf des gewählten Lösers
    center: np.ndarray                 # Ergebnis des Lösers
    reference: np.ndarray              # beste bekannte Mitte zum Exponenten p
    reference_on_data: bool | None     # nur p < 1: liegt die beste gefundene Mitte auf einem Kunden?
    solver_gap: float                  # Abstand des Lösers zur Referenz in km
    cost_p: float                      # Σ d^p der Referenz (bzw. längster Weg für p = ∞)
    costs: np.ndarray                  # Kosten der Referenz nach q = 1, 2, ∞
    optima: np.ndarray                 # kleinste Kosten nach q = 1, 2, ∞
    regrets: np.ndarray                # Regret der Referenz nach q = 1, 2, ∞ in Prozent
    nearest: float                     # Abstand der Referenz zum nächsten Kunden
    path_p: tuple
    path: np.ndarray                   # c(p) für p = PATH_P und ∞, Form (Stützstellen, 2)


def reference_center(points: np.ndarray, p: float, seed: int) -> tuple[np.ndarray, bool | None]:
    """Beste bekannte Mitte zum Exponenten p (Newton/Vardi–Zhang/Welzl/Schwerpunkt; für p < 1 `best_below_one` mit Zufallsstarts) und, für p < 1, ob sie auf einem Kunden liegt."""
    if p < 1.0:
        r = K.best_below_one(points, p, seed, C.STARTS_BELOW_ONE)
        return r.center, r.on_data
    return K.lp_center(points, p, seed), None


def analyse(settings: Settings) -> Analysis:
    points = build_points(settings)
    optimum = RG.optimal_centers(points, settings.seed)
    solve = run_solver(settings, points)
    ref, on_data = reference_center(points, settings.p, settings.seed)
    costs = np.array([K.lp_cost(points, ref, q) for q in RG.Q_VALUES])
    optima = np.array([K.lp_cost(points, optimum[q], q) for q in RG.Q_VALUES])
    path_p = C.PATH_P + (math.inf,)
    path = np.array([optimum[p] if p in optimum else K.lp_center(points, p, settings.seed) for p in path_p])
    p = settings.p
    return Analysis(
        settings, points, optimum[2.0], optimum[1.0], optimum[math.inf], K.one_center(points, settings.seed)[1], solve, solve.center, ref, on_data,
        float(np.linalg.norm(solve.center - ref)), K.lp_cost(points, ref, p), costs, optima, np.array([RG.regret_percent(c, o) for c, o in zip(costs, optima)]),
        float(np.linalg.norm(points - ref, axis=1).min()), path_p, path,
    )


# ---------------------------------------------------------------- Verdikte der Karte

def verdict_regret(a: Analysis) -> tuple[str, str]:
    """Kernfrage: Was kostet die gewählte Mitte in den drei Kriterien gegenüber dem jeweils besten Punkt?"""
    p = a.settings.p
    if p < 1.0:
        where = "auf einem Kunden" if a.reference_on_data else "im Innern zwischen den Kunden (ein weiteres lokales Minimum)"
        return "info", (f"Für p = {fmt_p(p)} < 1 liegt das beste gefundene Minimum **{where}**; der nächste Kunde ist {de(a.nearest, 2)} km entfernt. Gegenüber dem besten Punkt je Kriterium "
                        f"kostet die Mitte {de(a.regrets[0], 1)} % mehr Summe der Wege, {de(a.regrets[1], 1)} % mehr Quadratsumme und {de(a.regrets[2], 1)} % mehr längsten Weg.")
    worst = int(np.argmax(a.regrets))
    names = ("Summe der Wege", "Quadratsumme", "längster Weg")
    chosen = C.CRITERIA.index(a.settings.criterion)
    text = (f"Nach Ihrem Kriterium „{a.settings.criterion}“ liegt die Mitte für p = {fmt_p(p)} **{de(a.regrets[chosen], 1)} %** über dem Optimum (der besten Mitte dafür, p = {fmt_p(C.CRITERION_Q[chosen])}). "
            f"Über alle drei Kriterien kostet sie gegenüber dem jeweils besten Punkt {de(a.regrets[0], 1)} % mehr Summe der Wege, {de(a.regrets[1], 1)} % mehr Quadratsumme "
            f"und {de(a.regrets[2], 1)} % mehr längsten Weg; am teuersten wird sie nach dem Kriterium „{names[worst]}“.")
    peak = float(a.regrets.max())
    return ("success" if peak < 1.0 else "info" if peak < 10.0 else "warning"), text


def verdict_solver(a: Analysis) -> tuple[str, str]:
    """Verdikt zum gewählten Löser: ausgeführte Rechenart, Abbruch oder Abweichung von der besten bekannten Mitte."""
    s, solve = a.settings, a.solve
    p = s.p
    if p == math.inf:
        return "info", "Für p = ∞ gibt es keine Iteration: Der kleinste umschließende Kreis wird direkt aus höchstens drei Kunden berechnet (Welzl)."
    if p == 2.0:
        return "info", "Für p = 2 ist die Mitte der Schwerpunkt: eine Formel, keine Iteration."
    if p == 1.0:
        return "info", f"Für p = 1 rechnet der Vardi–Zhang-Löser aus Stück 5 der Reihe ({solve.iterations} Schritte, Status {solve.status}); er ist hier eine Stelle auf dem Pfad."
    if p < 1.0:
        if a.solver_gap > 1e-6:
            return "warning", (f"Die Gewichtsiteration aus einem einzelnen Start endet nach {solve.iterations} Schritten {de(a.solver_gap, 2)} km von der besten gefundenen Mitte entfernt: "
                               "Jeder Kunde ist ein lokales Minimum, ein Start findet nicht unbedingt das beste.")
        return "success", f"Die Gewichtsiteration endet nach {solve.iterations} Schritten in der besten gefundenen Mitte."
    if solve.status == "maxit":
        return "error", (f"Die Gewichtsiteration erreicht die **Grenze von {C.IRLS_MAXIT_LIVE} Schritten** ohne zu konvergieren; sie liegt {de(a.solver_gap, 2)} km von der Mitte entfernt. "
                         "Das ist ein Ergebnis, kein Fehler: ohne Dämpfung konvergiert dieses Verfahren für größere p nicht (siehe E7).")
    if solve.status == "stillstand":
        return "warning", f"Newton findet nach {solve.iterations} Schritten keinen Abstieg mehr; Abstand zur Mitte {de(a.solver_gap, 6)} km."
    if solve.status == "divergent":
        return "error", "Die Gewichtsiteration läuft in einen Zahlenüberlauf."
    return "success", f"{s.solver} konvergiert nach {solve.iterations} Schritten, Abstand zur Mitte {de(a.solver_gap, 6)} km."


# ---------------------------------------------------------------- Verdikte der Experimente

def unjp(v) -> float:
    """Exponent aus der Ergebnisdatei: der Text "inf" ist ∞."""
    return math.inf if v == "inf" else float(v)


def verdict_e1_example(rows: list[dict]) -> tuple[str, str]:
    """E1, Zahlenbeispiel (Werte 1 bis 9 und 40): p = 1 ein Intervall, die Mitte läuft nur langsam zur Mitte der Spannweite."""
    at = {unjp(r["p"]): r for r in rows}
    first, mid, far, last = at[1.0], at[8.0], at[32.0], at[math.inf]
    mid_c, far_c = (mid["lo"] + mid["hi"]) / 2, (far["lo"] + far["hi"]) / 2
    return "info", (f"Bei den Werten 1 bis 9 und 40 ist die Mitte für p = 1 das **Intervall {de(first['lo'], 0)} bis {de(first['hi'], 0)}**; für p = 8 liegt sie bei {de(mid_c, 2)}, "
                    f"für p = 32 bei {de(far_c, 2)} und für p = ∞ bei {de(last['lo'], 2)}. Der Pfad läuft stetig, aber langsam: Bei p = 32 fehlen noch {de(last['lo'] - far_c, 2)} Einheiten.")


def verdict_e1_axis(normal: dict, lognormal: dict, two: dict) -> tuple[str, str]:
    """E1, Achse: wie oft läuft der Pfad von p = 1 bis ∞ ohne Rückschritt? (Zellen für dasselbe n)"""
    return "warning", (f"Der Pfad ist **stetig, aber in den meisten Layouts nicht monoton**: Bei {normal['n']} normalverteilten Werten läuft er nur in {pct(normal['strict'][0], 0)} % der Instanzen von p = 1 bis ∞ ohne Rückschritt, "
                       f"bei zwei Häufungen in {pct(two['strict'][0], 0)} % (bis p = 8: {pct(two['strict_to_8'][0], 0)} %), bei lognormalen Werten in {pct(lognormal['strict'][0], 0)} %. "
                       f"Die Rückschritte sind klein: im Mittel {de(100 * normal['back_mean'], 2)} % der Spannweite (Normal), höchstens {de(100 * normal['back_max'], 1)} % in der ungünstigsten Instanz.")


def verdict_e1_plane(normal: dict, outpost: dict) -> tuple[str, str]:
    """E1, Ebene: wächst die Summe der Wege mit p, fällt der längste Weg? (Zellen für dasselbe n)"""
    worst = max(normal["sum_back_max_pct"], normal["long_back_max_pct"], outpost["sum_back_max_pct"], outpost["long_back_max_pct"])
    return "warning", (f"Auch in der Ebene: Bei {normal['n']} normalverteilten Kunden wächst die Summe der Wege nur in {pct(normal['sum_strict'][0], 0)} % der Instanzen mit p, und der längste Weg fällt nur in "
                       f"{pct(normal['long_strict'][0], 0)} % ohne Rückschritt; mit einem Außenposten sind es {pct(outpost['sum_strict'][0], 0)} % und {pct(outpost['long_strict'][0], 0)} %. "
                       f"Der größte Rückschritt beträgt {de(worst, 1)} % des Wertes.")


def verdict_e1_approach(cells: list[dict]) -> tuple[str, str]:
    """E1, Annäherung: wie weit ist c(p) vom 1-Center entfernt (in Radien)?"""
    def span(p):
        values = [next(r for r in c["rows"] if r["p"] == p)["dist_radius"] for c in cells]
        return min(values), max(values)
    lo8, hi8 = span(8.0)
    lo32, hi32 = span(32.0)
    return "info", (f"Das 1-Center erreicht man nur mit großem p: Bei p = 8 liegt c(p) noch {pct(lo8, 0)} bis {pct(hi8, 0)} % eines Radius vom 1-Center entfernt, bei p = 32 noch {pct(lo32, 1)} bis {pct(hi32, 1)} % "
                    f"(Mittel über Instanzen, je nach Layout; n = {cells[0]['n']}).")


def verdict_e2(outpost: dict, normal: dict) -> tuple[str, str]:
    """E2: die Regret-Matrix ist asymmetrisch (Zellen mit demselben n)."""
    m, nm = outpost["mean"], normal["mean"]
    return "warning", (f"Mit einem Außenposten (n = {outpost['n']}) zahlt, wer nach p = ∞ wählt und nach der Quadratsumme bewertet, **{de(m[2][1], 0)} %** Mehrkosten; wer nach p = 2 wählt und nach dem "
                       f"längsten Weg bewertet, {de(m[1][2], 0)} %. Ohne Außenposten (Normal, n = {normal['n']}) sind es {de(nm[2][1], 0)} % und {de(nm[1][2], 0)} %: Die Matrix ist nicht symmetrisch, "
                       "und wie stark sie es ist, hängt vom Layout ab.")


def verdict_e3(rows: list[dict]) -> tuple[str, str]:
    """E3: Abstand des Außenpostens; Regret von p = 1, bewertet nach dem längsten Weg."""
    at = {r["a"]: r["mean"][0][2] for r in rows}
    return "info", (f"Der Regret von p = 1, bewertet nach dem längsten Weg, liegt bei a = 2 noch bei {de(at[2.0], 1)} %, bei a = 3 bei {de(at[3.0], 1)} % und bei a = 6 bei {de(at[6.0], 1)} %: "
                    "Zwischen a = 2 und a = 3 kippt das 1-Center (bei 30 normalverteilten Kunden um den Ursprung; eine feinere Bestimmung der Schwelle ist nicht untersucht).")


def verdict_e4(cells: list[dict]) -> tuple[str, str]:
    """E4: bestes festes p je Layout (Zellen mit demselben n, in der Reihenfolge der Layouts)."""
    parts = ", ".join(f"{c['layout']} p = {fmt_p(unjp(c['best_p']))} ({de(c['best_value'], 1)} %)" for c in cells)
    p2 = {c["layout"]: c["mean"][list(c["grid"]).index(2.0)] for c in cells}
    return "info", (f"Das beste feste p hängt vom Layout ab (n = {cells[0]['n']}; in Klammern der mittlere größte Regret über die drei Kriterien): {parts}. "
                    f"Das Mittel (p = 2) ist kein Kompromiss: Sein größter Regret liegt bei {de(p2['Normal'], 1)} % (Normal) bzw. {de(p2['Außenposten'], 1)} % (Außenposten).")


def verdict_e5(plane: dict, axis: dict) -> tuple[str, str]:
    """E5: leere Zone in der Ebene (eine Zelle aus Anteil und Abstand) und auf der Achse (eine Zelle aus Anteil)."""
    rows = {unjp(r["p"]): r["mean"] for r in plane["rows"]}
    axis_rows = {unjp(r["p"]): r["mean"] for r in axis["rows"]}
    return "warning", (f"Bei zwei Häufungen im Abstand {plane['sep']:g} km mit {pct(plane['share'], 0)} % in der großen liegt das Mittel (p = 2) in {pct(rows[2.0], 0)} % der Instanzen in einer leeren Zone "
                       f"(nächster Kunde weiter als 1,5 km), der geometrische Median (p = 1) in {pct(rows[1.0], 0)} %, das 1-Center in {pct(rows[math.inf], 0)} %. Auf der Achse "
                       f"(Anteil {pct(axis['share'], 0)} %) sind es für p = 2 {pct(axis_rows[2.0], 0)} % und für p = 4 {pct(axis_rows[4.0], 0)} %.")


def verdict_e6_density(cell: dict) -> tuple[str, str]:
    """E6, Dichte gegen Masse (eine Zelle: Streuung der großen Häufung)."""
    at = {r["p"]: r["mean"] for r in cell["rows"]}
    return "success", (f"60 % der Werte streuen mit {cell['s_big']:g} um 0, 40 % liegen dicht (Streuung 0,5) um 8. Die Mitte liegt in der kleinen, dichten Häufung in {pct(at[1.0], 0)} % der Instanzen bei p = 1, "
                       f"in {pct(at[0.7], 0)} % bei p = 0,7, in {pct(at[0.5], 0)} % bei p = 0,5 und in {pct(at[0.3], 0)} % bei p = 0,3: Mit p gegen 0 wählt die Mitte die dichteste Stelle, auch gegen die Mehrheit der Werte, sofern die Mehrheit weit genug streut (bei Streuung 0,5 und 1 geschieht das für kein p).")


def verdict_e6_start(rows: list[dict]) -> tuple[str, str]:
    """E6, Start auf der Achse: Anteil der Zufallsstarts, die im besten Minimum enden (B) bzw. in dessen Häufung (A)."""
    at = {r["p"]: r for r in rows}
    return "warning", (f"Zufallsstarts finden das beste Minimum nur teilweise: Bei p = 0,7 enden {pct(at[0.7]['B'][0], 0)} % der Starts innerhalb 0,2 davon, bei p = 0,5 {pct(at[0.5]['B'][0], 0)} %, bei p = 0,2 {pct(at[0.2]['B'][0], 0)} %; "
                       f"in der richtigen Häufung enden {pct(at[0.5]['A'][0], 0)} % bzw. {pct(at[0.2]['A'][0], 0)} % (p = 0,5 bzw. 0,2). Jeder Kunde ist ein lokales Minimum: bei p = 0,2 gibt es im Mittel "
                       f"{at[0.2]['distinct'][0]:.0f} verschiedene Enden bei {at[0.2]['starts']} Starts.")


def verdict_e6_plane(cells: list[dict]) -> tuple[str, str]:
    """E6, Ebene: Anteil der Instanzen, in denen das beste Minimum auf einem Kunden liegt (Zellen: Normal mit n = 5, 10, 30)."""
    def at(cell, p):
        return next(r["mean"] for r in cell["rows"] if r["p"] == p)
    return "info", (f"In der Ebene liegt das beste gefundene Minimum für p = 0,5 in {pct(min(at(c, 0.5) for c in cells), 0)} bis {pct(max(at(c, 0.5) for c in cells), 0)} % der Instanzen auf einem Kunden, "
                    f"für p = 0,7 in {pct(min(at(c, 0.7) for c in cells), 0)} bis {pct(max(at(c, 0.7) for c in cells), 0)} %, für p = 0,9 nur in {pct(min(at(c, 0.9) for c in cells), 0)} bis {pct(max(at(c, 0.9) for c in cells), 0)} %: "
                    "Dazwischen liegt das Optimum oft im Innern, wie bei p = 1.")


def verdict_e7(normal: dict, two: dict) -> tuple[str, str]:
    """E7: Gewichtsiteration ohne und mit Dämpfung (Zellen für beide Layouts)."""
    def row(cell, p):
        return next(r for r in cell["rows"] if r["p"] == p)
    n4, n8, t3 = row(normal, 4.0), row(normal, 8.0), row(two, 3.0)
    return "error", (f"Ohne Dämpfung konvergiert die Gewichtsiteration bei p = 4 in {pct(n4['plain']['converged'], 0)} % der Läufe (Normal; Grenze {normal['maxit']} Schritte) und der Zielwert steigt in {pct(n4['plain']['rises'], 0)} %; "
                     f"bei p = 3 mit zwei Häufungen steigt er in {pct(t3['plain']['rises'], 0)} % und es braucht im Median {t3['plain']['median_its']:.0f} Schritte. Mit Dämpfung 1/(p − 1) konvergiert sie in "
                     f"{pct(n8['damped']['converged'], 0)} % der Läufe bei p = 8 (Median {n8['damped']['median_its']:.0f} Schritte; Newton: {n8['newton_median_its']:.0f}).")
