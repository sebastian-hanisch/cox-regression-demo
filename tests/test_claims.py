"""Jede Zahl der README steht in einer Messung (`tools/measure_cox_result.json`, eingefroren, feste Zufalls-Seeds) oder in einem Test.

Dieser Test prüft dreierlei:
  1. Die Tabellen der README stehen wörtlich so in der README, wie `tools/measure_cox.tables()` sie aus der eingefrorenen JSON-Datei bildet.
  2. Die Prosa-Zahlen der README (Anteile, Zählungen, Spannen) folgen aus denselben Zahlen.
  3. Ausgewählte Zellen werden mit dem Code der Demo neu gerechnet und stimmen mit der eingefrorenen Datei überein (Band 1e-6; Zählungen exakt).
Die übrigen Zellen sind nur eingefroren (die README sagt das in der Tabelle „Zahl → Test“)."""

import json
from pathlib import Path

import pytest

import cxr_constants as C
import cxr_evaluation as ev
import cxr_scenario as S
import measure_cox as MC

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DATA = json.loads((ROOT / "tools" / "measure_cox_result.json").read_text(encoding="utf-8"))
TOL = 1e-6


def test_alle_tabellen_der_readme_stehen_woertlich_in_der_readme():
    for name, table in MC.tables(DATA).items():
        assert table in README, name


def test_readme_zahlen_der_standardwelt():
    s = DATA["standard"]
    assert (s["n"], s["window"], s["beta"], s["shape"], s["seed"], s["reps"]) == (300, 12, 0.7, 1.5, 35, 200)
    assert ev.pct(s["censored"], 1) == "54,4 %" and "54,4 % zensiert" in README
    assert ev.num(s["cox_mean"], 3) == "0,703" and ev.num(s["cox_se_of_mean"], 3) == "0,007"
    assert ev.num(s["ols_all_mean"], 2) == "−0,21" and ev.num(s["truth_ols"], 2) == "−0,47"
    assert ev.pct(s["ols_all_mean"] / s["truth_ols"]) == "44 %" and ev.pct(s["ols_events_mean"] / s["truth_ols"]) == "46 %"
    assert "bei 44 % der Wahrheit" in README and "0,70 gegen 0,70" in README


def test_stabilitaet_und_zusammengefasste_ueberdeckung():
    cells = [DATA["standard"]] + DATA["stability"]
    assert [c["seed"] for c in cells] == [35, 1, 2, 3, 4, 5] and all(c["reps"] == 200 and c["n"] == 300 for c in cells)
    hits = sum(c["coverage_hits"] for c in cells)
    total = sum(c["n_ok"] for c in cells)
    assert (hits, total) == (1129, 1200) and ev.pct(hits / total, 1) == "94,1 %"
    shares = [c["coverage_hits"] / c["n_ok"] for c in cells]
    assert ev.pct(min(shares), 1) == "92,0 %" and ev.pct(max(shares), 1) == "96,5 %"
    assert "1129 von 1200 Läufen (94,1 %)" in README and "zwischen 92,0 % und 96,5 %" in README and "von 92,0 % bis 96,5 %" in README
    means = [c["cox_mean"] for c in cells]
    assert ev.num(min(means), 2) == "0,69" and ev.num(max(means), 2) == "0,72"
    assert {ev.num(c["ols_all_mean"], 2) for c in cells} == {"−0,21"}


def test_ueberdeckung_gegen_den_plan():
    by_n = {c["n"]: c for c in DATA["by_n"]}
    assert [ev.pct(by_n[n]["coverage_hits"] / by_n[n]["n_ok"], 1) for n in (50, 100, 300)] == ["95,5 %", "94,5 %", "96,5 %"]
    assert "191 / 189 / 193 von 200 (95,5 / 94,5 / 96,5 %)" in README


def test_kleines_n_und_standardfehler():
    by_n = {c["n"]: c for c in DATA["by_n"]}
    assert by_n[10]["n_no_fit"] == 9 and by_n[10]["status_counts"] == {"ok": 191, "trennung": 9}
    assert ev.num(by_n[10]["cox_mean"]) == "1,03" and ev.num(by_n[20]["cox_mean"]) == "0,85" and by_n[20]["n_no_fit"] == 0
    for n in (50, 100, 300, 600):
        c = by_n[n]
        assert abs(c["mean_se"] / c["cox_sd"] - 1) <= 0.05, n          # „höchstens 5 %“
    assert "(0,255 gegen 0,268 bei n = 50, 0,172 gegen 0,167 bei n = 100, 0,096 gegen 0,096 bei n = 300, 0,067 gegen 0,065 bei n = 600)" in README
    assert [ev.num(by_n[n]["mean_se"], 3) for n in (50, 100, 300, 600)] == ["0,255", "0,172", "0,096", "0,067"]
    assert [ev.num(by_n[n]["cox_sd"], 3) for n in (50, 100, 300, 600)] == ["0,268", "0,167", "0,096", "0,065"]


def test_fenster_je_kuerzer_desto_staerker_verzerrt_und_cox_bleibt():
    cells = DATA["by_window"]
    ratios = [c["ols_all_mean"] / c["truth_ols"] for c in cells]
    assert [c["window"] for c in cells] == list(C.WINDOW_OPTIONS)
    assert all(a < b for a, b in zip(ratios, ratios[1:]))              # monoton: längeres Fenster, kleinere Verzerrung
    assert [c["censored"] for c in cells] == sorted((c["censored"] for c in cells), reverse=True)
    assert {ev.num(c["cox_mean"], 2) for c in cells} == {"0,70", "0,71"}   # „0,70 bis 0,71“
    assert ev.pct(ratios[0]) == "14 %" and ev.pct(ratios[-1]) == "99 %" and ev.pct(cells[-1]["censored"], 1) == "0,8 %"


def test_andere_grundformen_ohne_wirkung_und_handwerte():
    shapes = {c["shape"]: c for c in DATA["by_shape"]}
    assert [ev.pct(shapes[k]["ols_all_mean"] / shapes[k]["truth_ols"]) for k in (1.0, 1.5, 2.5)] == ["48 %", "44 %", "40 %"]
    z = DATA["no_effect"]
    assert (ev.num(z["cox_mean"], 3), ev.num(z["cox_se_of_mean"], 3), ev.num(z["ols_all_mean"], 3), ev.num(z["ols_events_mean"], 3), z["coverage_hits"]) == ("0,006", "0,006", "0,005", "0,007", 186)
    h = DATA["hand"]
    assert h["score0"] == pytest.approx(2 / 3, abs=1e-14) and h["info0"] == pytest.approx(13 / 18, abs=1e-14) and h["newton1"] == pytest.approx(12 / 13, abs=1e-14)
    assert ev.num(h["beta_hat"], 4) == "0,9406" and ev.num(h["hazard_ratio"], 3) == "2,562" and ev.num(h["se"], 2) == "1,24" and ev.num(h["loglik0"], 3) == "−3,178"


def test_kurven_prosazahlen():
    c = DATA["curves"]
    assert ev.num(100 * c["truth_gap_median"], 1) == "7,2" and ev.num(100 * c["km_truth_gap_median"], 1) == "11,0"
    assert "(7,2 gegen 11,0 Prozentpunkte bei n = 300)" in README
    assert c["truth_gap_median"] < c["km_truth_gap_median"]


# --- Neuberechnung ausgewählter Zellen mit dem Code der Demo ---------------------------------------------------------------------------------------


def _same(cell: dict, frozen: dict):
    for key, value in frozen.items():
        if isinstance(value, dict):
            assert cell[key] == value, key
        elif isinstance(value, float):
            assert cell[key] == pytest.approx(value, abs=TOL), key
        else:
            assert cell[key] == value, key


@pytest.mark.parametrize("name,world", [
    ("standard", S.World()),
    ("stability0", S.World(seed=1)),
    ("by_n10", S.World(n=10)),
    ("by_n50", S.World(n=50)),
    ("window4", S.World(window=4)),
    ("window1000", S.World(window=1000)),
    ("shape2.5", S.World(shape=2.5)),
    ("no_effect", S.World(beta=0.0)),
])
def test_zellen_werden_mit_dem_code_der_demo_neu_gerechnet(name, world):
    frozen = {
        "standard": DATA["standard"], "stability0": DATA["stability"][0], "by_n10": DATA["by_n"][0], "by_n50": DATA["by_n"][2],
        "window4": DATA["by_window"][0], "window1000": DATA["by_window"][-1], "shape2.5": DATA["by_shape"][2], "no_effect": DATA["no_effect"],
    }[name]
    _same(MC.cell(world), frozen)


def test_experiment_drei_und_handbeispiel_werden_neu_gerechnet():
    _same(MC.curve_cell(S.World()), DATA["curves"])
    _same(MC.hand_cell(), DATA["hand"])


# --- Der Annahmen-Text steht in App und README gleich ---------------------------------------------------------------------------------------------


def test_annahmen_text_der_app_steht_woertlich_in_der_readme():
    assert C.ASSUMPTIONS_MARKDOWN.strip() in README


def test_readme_nennt_die_unabhaengigkeitsannahme_am_anfang():
    assert README.index("Die Zensierung ist unabhängig vom Ereignis") < README.index("## Warum dieses Problem")
