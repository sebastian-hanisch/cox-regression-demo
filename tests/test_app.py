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
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen dieser Demo (der Anfang der Datei ist die Vorlage tools/templates/test_app_smoke.py) ------------------------------------------------
# Regler dieser App: Slider (Wirkung β, Form k, Container n, β zum Ausprobieren), Auswahlregler (Beobachtungsfenster), Auswahlliste (Muster des Handbeispiels),
# Zahlenfeld (Zufalls-Seed). Randfälle der Cox-Anpassung (Trennung, konstantes Merkmal, kleinste Stichprobe, β = 0) müssen als Meldung erscheinen, nie als Fehler.
import pytest  # noqa: E402

import cxr_constants as C  # noqa: E402
from cxr_scenario import PATTERNS  # noqa: E402


def _texts(at) -> str:
    return " ".join(m.value for kind in (at.success, at.info, at.warning, at.error, at.markdown, at.caption) for m in kind)


@pytest.mark.parametrize("window", C.WINDOW_OPTIONS)
def test_every_observation_window_runs(window):
    at = _fresh_app()
    at.select_slider(key="window_select").set_value(window).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_every_pattern_of_the_hand_example_runs_and_reports_its_status(pattern):
    at = _fresh_app()
    at.selectbox(key="pattern_select").set_value(pattern).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    if "Trennung" in pattern:
        assert any("Kein Schätzer" in w.value and "Perfekte Trennung" in w.value for w in at.warning)   # das Konvergenzflag erscheint als Meldung
        assert not any("β̂ = " in s.value for s in at.success if "Maximum" in s.value)
    elif "Konstantes" in pattern:
        assert any("Kein Schätzer" in w.value and "Information über β ist 0" in w.value for w in at.warning)
    else:
        assert any("Maximum bei **β̂ = " in s.value for s in at.success)


def test_extreme_combinations_of_all_controls_do_not_raise():
    """Alle Regler zugleich am Minimum bzw. am Maximum (Eckfall-Matrix): kleinste Stichprobe mit stärkstem Zensieren, größte mit längstem Fenster."""
    for n, window, beta, shape in ((C.N_MIN, 4, C.BETA_MAX, C.SHAPE_MAX), (C.N_MAX, 1000, C.BETA_MIN, C.SHAPE_MIN), (C.N_MIN, 1000, C.BETA_MIN, C.SHAPE_MIN), (C.N_MAX, 4, C.BETA_MAX, C.SHAPE_MAX)):
        at = _fresh_app()
        at.slider(key="n_slider").set_value(n)
        at.select_slider(key="window_select").set_value(window)
        at.slider(key="beta_slider").set_value(beta)
        at.slider(key="shape_slider").set_value(shape)
        at.run(timeout=TIMEOUT)
        assert not at.exception, ((n, window, beta, shape), [str(e) for e in at.exception])


def test_smallest_sample_reports_runs_without_estimate_in_words():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(C.N_MIN).run(timeout=TIMEOUT)
    assert not at.exception
    assert "keinen endlichen Cox-Schätzer" in _texts(at) and "perfekte Trennung" in _texts(at)


def test_seed_extremes_do_not_raise():
    for value in (0, C.SEED_MAX):
        at = _fresh_app()
        at.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (value, [str(e) for e in at.exception])


def test_the_independence_assumption_is_visible_in_the_intro_and_in_the_assumptions_expander():
    at = _fresh_app()
    assert any("Die Zensierung ist unabhängig vom Ereignis" in m.value for m in at.markdown)
    expander = next(e for e in at.expander if e.label == "⚠️ Annahmen, Grenzen und bekannte Probleme")
    assert any("Die Zensierung ist unabhängig vom Ereignis" in m.value for m in expander.markdown)
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
