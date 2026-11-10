"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import cxr_constants as C
import cxr_presets as P
import cxr_scenario as S

WORLD_KEYS = ("n", "window", "beta", "shape", "seed")


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_world():
    standard = C.PRESETS["Messung V6 (Standard)"]
    assert S.World(**{k: standard[k] for k in WORLD_KEYS}) == S.World()
    assert "pattern" not in standard


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_defaults_lie_inside_their_bounds_and_choices():
    for key, spec in P.SETTING_SPECS.items():
        assert P._clean(spec, spec.default) == spec.default, key


def test_permalink_values_are_clamped_and_unknown_choices_fall_back():
    assert P._clean(P.SETTING_SPECS["n_slider"], 10**6) == C.N_MAX
    assert P._clean(P.SETTING_SPECS["n_slider"], -5) == C.N_MIN
    assert P._clean(P.SETTING_SPECS["window_select"], 13) == C.DEFAULT_WINDOW          # 13 ist keine Auswahl
    assert P._clean(P.SETTING_SPECS["window_select"], 40) == 40
    assert P._clean(P.SETTING_SPECS["pattern_select"], "Unfug") == S.DEFAULT_PATTERN


def test_float_comparison_tolerates_slider_rounding():
    assert P._gleich(0.7000000000000001, 0.7) and not P._gleich(0.75, 0.7) and P._gleich(300, 300) and not P._gleich("a", "b")
