"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue          # gesperrt (Layout oder Auswahl braucht ihn nicht): die Eckfälle-Tests unten schalten ihn über das Layout frei
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_other_regler_with_options_at_each_end_does_not_raise():
    import math

    import lpm_constants as C
    for key, options in (("p_select", (C.P_OPTIONS[0], C.P_OPTIONS[-1], 1.0, 0.9, 1.25)), ("layout_select", C.LAYOUTS), ("criterion_select", C.CRITERIA), ("solver_select", C.SOLVERS),
                         ("start_select", C.STARTS)):
        for value in options:
            at = _fresh_app()
            if key != "p_select":
                at.sidebar.select_slider(key="p_select").set_value(3.0).run(timeout=TIMEOUT)      # schaltet Löser und Start frei
            widget = at.sidebar.select_slider(key=key) if key == "p_select" else at.sidebar.selectbox(key=key)
            widget.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])
    assert math.inf in C.P_OPTIONS


def test_corner_combinations_of_all_regler_do_not_raise():
    """Alle Minima und alle Maxima zugleich, dazu der Außenposten mit p = ∞ und zwei Häufungen mit p < 1 am Maximum von n."""
    import math

    import lpm_constants as C
    corners = [
        {"layout_select": "Normal", "n_slider": C.N_MIN, "p_select": C.P_OPTIONS[0], "outpost_slider": C.OUTPOST_MIN, "share_slider": C.SHARE_MIN, "sep_slider": C.SEP_MIN, "spread_slider": C.SPREAD_MIN},
        {"layout_select": "Zwei Häufungen", "n_slider": C.N_MAX, "p_select": math.inf, "outpost_slider": C.OUTPOST_MAX, "share_slider": C.SHARE_MAX, "sep_slider": C.SEP_MAX, "spread_slider": C.SPREAD_MAX},
        {"layout_select": "Außenposten", "n_slider": C.N_MAX, "p_select": math.inf, "outpost_slider": C.OUTPOST_MAX},
        {"layout_select": "Zwei Häufungen", "n_slider": C.N_MAX, "p_select": 0.4, "share_slider": 60, "spread_slider": C.SPREAD_MAX},
        {"layout_select": "Quadrat", "n_slider": C.N_MIN, "p_select": C.P_FINITE_MAX, "solver_select": C.SOLVERS[2]},
    ]
    for corner in corners:
        at = _fresh_app()
        at.sidebar.select_slider(key="p_select").set_value(corner["p_select"]).run(timeout=TIMEOUT)
        at.sidebar.selectbox(key="layout_select").set_value(corner["layout_select"]).run(timeout=TIMEOUT)       # schaltet die Regler des Layouts frei
        for key, value in corner.items():
            widget = at.sidebar.select_slider(key=key) if key == "p_select" else at.sidebar.selectbox(key=key) if key.endswith("_select") else at.sidebar.slider(key=key)
            if not widget.disabled:          # Regler, die das Layout oder p nicht braucht, sind gesperrt (siehe test_the_conditional_regler_are_disabled_where_they_do_nothing)
                widget.set_value(value)
        at.run(timeout=TIMEOUT)
        assert not at.exception, (corner, [str(e) for e in at.exception])


def test_the_conditional_regler_are_disabled_where_they_do_nothing():
    import lpm_constants as C
    at = _fresh_app()
    assert at.sidebar.slider(key="outpost_slider").disabled and at.sidebar.slider(key="share_slider").disabled     # Normal: weder Außenposten noch Häufungen
    assert at.sidebar.selectbox(key="solver_select").disabled and at.sidebar.selectbox(key="start_select").disabled      # p = 2: Schwerpunkt, keine Iteration
    at.sidebar.select_slider(key="p_select").set_value(3.0)
    at.sidebar.selectbox(key="layout_select").set_value("Außenposten").run(timeout=TIMEOUT)
    assert not at.sidebar.slider(key="outpost_slider").disabled and at.sidebar.slider(key="share_slider").disabled
    assert not at.sidebar.selectbox(key="solver_select").disabled and not at.sidebar.selectbox(key="start_select").disabled
    at.sidebar.select_slider(key="p_select").set_value(0.5).run(timeout=TIMEOUT)
    assert at.sidebar.selectbox(key="solver_select").disabled and not at.sidebar.selectbox(key="start_select").disabled
    assert C.SOLVERS


def test_every_live_button_produces_its_result():
    at = _fresh_app()
    charts_before = len(at.get("plotly_chart"))
    for key in ("e2_live", "e4_live", "e7_live", "pdf_make"):
        at.button(key=key).click().run(timeout=TIMEOUT)
        assert not at.exception, (key, [str(e) for e in at.exception])
    assert any("Nachgemessen (Normal, n = 30)" in i.value for i in at.info)                                    # E4
    assert any("Schritte" in str(df.value.to_dict()) for df in at.dataframe)                                 # E7: Tabelle mit den Läufen der Instanz
    assert at.get("download_button") and len(at.get("plotly_chart")) == charts_before + 2          # E2 zeigt die nachgemessene Matrix, E4 das nachgemessene Diagramm
