"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und jede Zahl der Hilfetexte stimmt mit der Rechnung überein (Hilfetexte nach der Messung, DEMO-PLAYBOOK §4)."""

import math
from dataclasses import replace

import pytest

import lpm_constants as C
import lpm_evaluation as V
import lpm_presets as P

STANDARD = next(iter(C.PRESETS))


def _settings(name):
    return V.Settings(**C.PRESETS[name])


def _help(name):
    return C.PRESET_HELP[name]


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert V.Settings(**C.PRESETS[STANDARD]) == V.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_standard_is_the_centroid_with_no_iteration():
    a = V.analyse(_settings(STANDARD))
    assert a.reference == pytest.approx(a.centroid) and a.solve.iterations == 0 and a.regrets[1] == pytest.approx(0.0, abs=1e-9)


def test_outpost_preset_numbers_in_the_help():
    a = V.analyse(_settings("Ein Außenposten (a = 6)"))
    assert a.points[0] == pytest.approx([6.0, 6.0]) and len(a.points) == 30
    assert f"{V.de(a.regrets[0], 1)} % über dem Optimum" in _help("Ein Außenposten (a = 6)") and f"{V.de(a.regrets[2], 1)} % über dem 1-Center" in _help("Ein Außenposten (a = 6)")
    assert V.criterion_q(_settings("Ein Außenposten (a = 6)").criterion) == math.inf


def test_two_areas_preset_puts_the_mean_into_the_empty_zone_and_the_median_into_the_large_cluster():
    name = "Zwei Liefergebiete"
    a = V.analyse(_settings(name))
    assert a.nearest > 1.5 and a.median[0] < 2.0 < a.centroid[0]
    assert f"bei x = {V.de(a.centroid[0], 2)}" in _help(name) and f"{V.de(a.nearest, 2)} km vom nächsten Kunden" in _help(name) and f"bei x = {V.de(a.median[0], 2)} in der großen" in _help(name)


def test_density_preset_jumps_to_the_dense_cluster_and_a_single_start_misses_it():
    name = "Dichte gegen Masse"
    s = _settings(name)
    a = V.analyse(s)
    assert a.reference_on_data and a.reference[0] > 6.0 and a.median[0] < 4.0
    assert a.solver_gap > 1.0                       # der einzelne Start vom Schwerpunkt endet woanders
    h = _help(name)
    assert f"Bei p = 1 liegt die Mitte bei x = {V.de(a.median[0], 2)}" in h and f"(x = {V.de(a.reference[0], 2)})" in h and f"Er endet {V.de(a.solver_gap, 1)} km davon entfernt" in h


def test_compromise_preset_lowers_the_worst_regret_against_p_two():
    name = "Kompromiss-p"
    s = _settings(name)
    a3, a2 = V.analyse(s), V.analyse(replace(s, p=2.0))
    assert a3.regrets.max() < a2.regrets.max()
    h = _help(name)
    assert f"von {V.de(a2.regrets.max(), 1)} % (p = 2) auf {V.de(a3.regrets.max(), 1)} %" in h
    assert f"Summe und Quadratsumme {V.de(a3.regrets[0], 1)} % und {V.de(a3.regrets[1], 1)} % mehr" in h


def test_iteration_preset_hits_the_limit_while_damping_and_newton_converge():
    name = "Die Iteration hängt"
    s = _settings(name)
    a = V.analyse(s)
    assert a.solve.status == "maxit" and a.solve.iterations == C.IRLS_MAXIT_LIVE == 3000 and 1.1 < a.solver_gap < 1.3
    damped, newton = V.analyse(replace(s, solver=C.SOLVERS[2])), V.analyse(replace(s, solver=C.SOLVERS[0]))
    assert damped.solve.status == newton.solve.status == "ok" and damped.solve.iterations < 50 and newton.solve.iterations < 10
    assert damped.solver_gap < 1e-6 and newton.solver_gap < 1e-6
    h = _help(name)
    assert "Nach 3000 Schritten" in h and "rund 1,2 km" in h and "unter 50 Schritte" in h and "unter 10 Schritte" in h


def test_every_layout_changes_the_regret_matrix_against_normal():
    """Nullspalten-Regel: Jedes Layout verändert mindestens einen Regret-Eintrag gegenüber Normal."""
    import lpm_regret as RG
    from lpm_scenario import make_points
    base = RG.regret_matrix(make_points("Normal", 30, 7))
    for layout in C.LAYOUTS[1:]:
        other = RG.regret_matrix(make_points(layout, 30, 7))
        assert abs(other - base).max() > 1.0, layout
