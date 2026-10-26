"""Die kopierten Bausteine (Entscheidung der Reihe: wörtlich kopieren, Quelle im Kopf der Datei): Vergleich mit dem Original, solange es im Portfolio-Ordner liegt."""

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("reduktionskette-demo", "red_scenario.py")
GEOMED_SOURCE = ("weiszfeld-demo", "wsz_geomed.py")
GEOMED_PARTS = ("COINCIDENT", "Result", "weighted_centroid", "_start_point", "vardi_zhang")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "lpm_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """Die Kopie läuft für sich: SplitMix64 mit Seed 0 liefert die bekannten ersten Werte der Konvention (Vigna)."""
    from lpm_rng import SplitMix64
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [16294208416658607535, 7960286522194355700, 487617019471545679]
    assert SplitMix64(7).below(10) == SplitMix64(7).below(10) < 10


def _segments(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in GEOMED_PARTS:
            out[name] = ast.get_source_segment(text, node)
    return out


def test_vardi_zhang_and_its_helpers_are_verbatim_copies_of_stueck_5():
    folder, original = GEOMED_SOURCE
    source = ROOT.parent / folder / original
    mine = _segments(ROOT / "lpm_geomed.py")
    assert set(mine) == set(GEOMED_PARTS)
    head = (ROOT / "lpm_geomed.py").read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith(f"# Wörtliche Kopie einzelner Bausteine von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    theirs = _segments(source)
    for name in GEOMED_PARTS:
        assert mine[name] == theirs[name], f"{name} weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_vardi_zhang_stops_on_a_customer_that_holds_the_location():
    """Die Kopie läuft für sich: der Kunde (0, 0) mit Gewicht 5 hält den Standort fest (Zugkraft der anderen √2 ≤ 5)."""
    from lpm_geomed import vardi_zhang
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
    r = vardi_zhang(pts, np.array([5.0, 1.0, 1.0]), start=[0.0, 0.0])
    assert r.status == "datenpunkt" and r.center == pytest.approx([0.0, 0.0])
