"""Vorgerechnete Messreihen (data/lpm_results.json, erzeugt von tools/make_results.py aus lpm_experiments.py): Laden und Abfragen.

Die Datei trägt die Prüfsumme der Quelldateien; `is_current()` vergleicht sie mit dem heutigen Stand (tests/test_results.py).
Unendlich steht in der Datei als Text "inf"; die Abfragen geben `math.inf` zurück."""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path

import lpm_constants as C

PATH = Path(os.environ.get("LPM_RESULTS_PATH") or Path(__file__).resolve().parent / "data" / C.RESULTS_FILE)


@functools.lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def meta() -> dict:
    return load()["meta"]


def is_current() -> bool:
    import lpm_experiments as X
    return meta()["erzeuger_hash"] == X.source_hash()


def p_of(value) -> float:
    """Exponent aus der Datei: der Text "inf" ist ∞."""
    return math.inf if value == "inf" else float(value)


def example() -> dict:
    return load()["m1a"]["example"]


def example_row(p: float) -> dict:
    (row,) = [r for r in example()["rows"] if p_of(r["p"]) == p]
    return row


def axis_cell(layout: str, n: int) -> dict:
    (row,) = [r for r in load()["m1a"]["axis"] if r["layout"] == layout and r["n"] == n]
    return row


def axis_cells() -> list[dict]:
    return load()["m1a"]["axis"]


def plane_cell(layout: str, n: int) -> dict:
    (row,) = [r for r in load()["m1b"]["plane"] if r["layout"] == layout and r["n"] == n]
    return row


def plane_cells() -> list[dict]:
    return load()["m1b"]["plane"]


def approach_cell(layout: str) -> dict:
    (row,) = [r for r in load()["m1b"]["approach"] if r["layout"] == layout]
    return row


def approach_row(layout: str, p: float) -> dict:
    (row,) = [r for r in approach_cell(layout)["rows"] if r["p"] == p]
    return row


def regret_cells() -> list[dict]:
    return load()["m2"]["raster"]


def regret_cell(layout: str, n: int) -> dict:
    (row,) = [r for r in regret_cells() if r["layout"] == layout and r["n"] == n]
    return row


def outpost_rows() -> list[dict]:
    return load()["m2"]["outpost"]


def outpost_row(a: float) -> dict:
    (row,) = [r for r in outpost_rows() if r["a"] == a]
    return row


def compromise_cells() -> list[dict]:
    return load()["m4"]["cells"]


def compromise_cell(layout: str, n: int) -> dict:
    (row,) = [r for r in compromise_cells() if r["layout"] == layout and r["n"] == n]
    return row


def empty_plane_cells() -> list[dict]:
    return load()["m5"]["plane"]


def empty_plane_cell(share: float, sep: float) -> dict:
    (row,) = [r for r in empty_plane_cells() if r["share"] == share and r["sep"] == sep]
    return row


def empty_plane_value(share: float, sep: float, p: float) -> float:
    (row,) = [r for r in empty_plane_cell(share, sep)["rows"] if p_of(r["p"]) == p]
    return row["mean"]


def empty_axis_cells() -> list[dict]:
    return load()["m5"]["axis"]


def empty_axis_value(share: float, p: float) -> float:
    (cell,) = [r for r in empty_axis_cells() if r["share"] == share]
    (row,) = [r for r in cell["rows"] if p_of(r["p"]) == p]
    return row["mean"]


def density_cells() -> list[dict]:
    return load()["m6a"]["density"]


def density_value(s_big: float, p: float) -> float:
    (cell,) = [r for r in density_cells() if r["s_big"] == s_big]
    (row,) = [r for r in cell["rows"] if r["p"] == p]
    return row["mean"]


def start_rows() -> list[dict]:
    return load()["m6a"]["start"]


def start_row(p: float) -> dict:
    (row,) = [r for r in start_rows() if r["p"] == p]
    return row


def stair_rows() -> list[dict]:
    return load()["m6a"]["stairs"]


def stair_row(layout: str, lo: float, step: float) -> dict:
    (row,) = [r for r in stair_rows() if r["layout"] == layout and r["lo"] == lo and r["step"] == step]
    return row


def plane_data_cells() -> list[dict]:
    return load()["m6b"]["plane_data"]


def plane_data_value(layout: str, n: int, p: float) -> float:
    (cell,) = [r for r in plane_data_cells() if r["layout"] == layout and r["n"] == n]
    (row,) = [r for r in cell["rows"] if r["p"] == p]
    return row["mean"]


def plane_density_cells() -> list[dict]:
    return load()["m6b"]["plane_density"]


def plane_density_value(spread: float, p: float) -> float:
    (cell,) = [r for r in plane_density_cells() if r["spread"] == spread]
    (row,) = [r for r in cell["rows"] if r["p"] == p]
    return row["mean"]


def plane_start_rows() -> list[dict]:
    return load()["m6b"]["plane_start"]


def plane_start_row(p: float) -> dict:
    (row,) = [r for r in plane_start_rows() if r["p"] == p]
    return row


def irls_cells() -> list[dict]:
    return load()["m7"]["cells"]


def irls_cell(layout: str) -> dict:
    (row,) = [r for r in irls_cells() if r["layout"] == layout]
    return row


def irls_row(layout: str, p: float) -> dict:
    (row,) = [r for r in irls_cell(layout)["rows"] if r["p"] == p]
    return row
