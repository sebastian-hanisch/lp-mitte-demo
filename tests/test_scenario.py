"""Szenario: der blockweise Strom stimmt mit der kopierten SplitMix64-Klasse überein, die Layouts haben die dokumentierte Form, alles ist deterministisch."""

import numpy as np
import pytest

import lpm_scenario as S
from lpm_rng import SplitMix64


def test_draws_raw_equals_the_splitmix64_stream():
    rng = SplitMix64(12345)
    expected = [rng.next() for _ in range(50)]
    assert [int(v) for v in S.Draws(12345).raw(50)] == expected
    d = S.Draws(12345)
    first, second = d.raw(20), d.raw(30)
    assert [int(v) for v in np.concatenate([first, second])] == expected             # Fortsetzung des Stroms über Blöcke
    assert len(S.Draws(1).raw(0)) == 0 and len(S.Draws(1).raw(-3)) == 0


def test_uniform_and_normal_have_the_right_moments_and_odd_sizes_work():
    u = S.Draws(4).uniform(200000)
    assert 0.0 <= u.min() and u.max() < 1.0 and u.mean() == pytest.approx(0.5, abs=0.005)
    z = S.Draws(4).normal(200000)
    assert z.mean() == pytest.approx(0.0, abs=0.01) and z.std() == pytest.approx(1.0, abs=0.01)
    assert S.Draws(4).normal(7).shape == (7,) and S.Draws(4).normal((3, 5)).shape == (3, 5)
    assert S.Draws(4).uniform((2, 3)).shape == (2, 3)
    first_pair = S.Draws(4).normal(2)
    u2 = S.Draws(4).uniform((2, 1))
    assert first_pair[0] == pytest.approx(np.sqrt(-2 * np.log(1 - u2[0, 0])) * np.cos(2 * np.pi * u2[1, 0]))


def test_cell_seed_is_stable_and_distinguishes_marks():
    assert S.cell_seed(11, "Normal", 30) == S.cell_seed(11, "Normal", 30)
    assert len({S.cell_seed(11, "Normal", 30), S.cell_seed(12, "Normal", 30), S.cell_seed(11, "Quadrat", 30), S.cell_seed(11, "Normal", 31)}) == 4
    assert S.cell_seed(11, 0.5) != S.cell_seed(11, 0.6)


def test_group_size_is_rounded_and_never_empty():
    assert S.group_size(0.7, 10) == 7 and S.group_size(0.7, 30) == 21 and S.group_size(0.6, 5) == 3 and S.group_size(0.5, 5) == 3
    assert S.group_size(0.95, 5) == 4 and S.group_size(0.5, 2) == 1 and S.group_size(0.0, 5) == 1


def test_layouts_have_the_documented_shape():
    d = S.Draws(3)
    assert S.layout_batch(d, "Normal", 4, 7).shape == (4, 7, 2)
    q = S.layout_batch(S.Draws(3), "Quadrat", 50, 20)
    assert q.min() >= -1.0 and q.max() < 1.0
    o = S.layout_batch(S.Draws(3), "Außenposten", 5, 10, outpost=9.0)
    assert (o[:, 0] == 9.0).all() and np.abs(o[:, 1:]).max() < 6
    zh = S.layout_batch(S.Draws(3), "Zwei Häufungen", 400, 10, share=0.7, sep=20.0, spread=3.0)
    assert zh[:, :7, 0].mean() == pytest.approx(0.0, abs=0.4) and zh[:, 7:, 0].mean() == pytest.approx(20.0, abs=0.4)
    assert zh[:, :7].std() == pytest.approx(3.0, rel=0.05) and zh[:, 7:, 1].std() == pytest.approx(1.0, rel=0.1)
    with pytest.raises(ValueError):
        S.layout_batch(S.Draws(1), "Kreis", 1, 5)


def test_make_points_is_deterministic_and_seed_dependent():
    a = S.make_points("Normal", 12, 5)
    assert a.shape == (12, 2) and (a == S.make_points("Normal", 12, 5)).all() and not (a == S.make_points("Normal", 12, 6)).all()


def test_axis_layouts():
    assert S.axis_batch(S.Draws(2), "Normal", 3, 9).shape == (3, 9)
    assert (S.axis_batch(S.Draws(2), "Lognormal", 50, 9) > 0).all()
    o = S.axis_batch(S.Draws(2), "Außenposten", 5, 9, outpost=6.0)
    assert (o[:, 0] == 6.0).all()
    z = S.axis_batch(S.Draws(2), "Zwei Häufungen", 500, 10, share=0.7, sep=6.0)
    assert z[:, :7].mean() == pytest.approx(0.0, abs=0.15) and z[:, 7:].mean() == pytest.approx(6.0, abs=0.15)
    with pytest.raises(ValueError):
        S.axis_batch(S.Draws(1), "Kreis", 1, 5)


def test_density_batch_two_scales():
    x = S.density_batch(S.Draws(8), 500, 10, s_big=3.0, share=0.6, sep=8.0, s_small=0.5)
    assert x[:, :6].std() == pytest.approx(3.0, rel=0.06) and x[:, 6:].std() == pytest.approx(0.5, rel=0.08) and x[:, 6:].mean() == pytest.approx(8.0, abs=0.1)
