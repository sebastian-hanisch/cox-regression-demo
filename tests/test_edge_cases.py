"""Randfälle der Cox-Anpassung und der Auswertung (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstände, alles zensiert, konstantes Merkmal, ungültige Zeiten.
Jede Zeile ohne Schätzer meldet einen Status; es gibt nie eine stille Zahl."""

import math

import numpy as np
import pytest

import cxr_constants as C
import cxr_core as K
import cxr_evaluation as ev
import cxr_model as M
import cxr_scenario as S


def test_ein_container_hat_keinen_schaetzer():
    assert M.fit_cox([2.0], [1], [0.5]).status == "singulaer"          # ein Merkmalswert: keine Information
    assert M.fit_cox([2.0], [0], [0.5]).status == "ohne_ereignis"


def test_genau_zwei_container():
    # x = 0, 1 mit Abholung des Containers mit dem größeren Merkmal zuerst: Trennung
    assert M.fit_cox([1.0, 2.0], [1, 1], [1.0, 0.0]).status == "trennung"
    assert M.fit_cox([1.0, 2.0], [1, 1], [0.0, 1.0]).status == "trennung"          # in die andere Richtung
    assert M.fit_cox([1.0, 2.0], [1, 0], [0.0, 1.0]).status == "trennung"          # ein Zensierter bleibt in der Risikomenge
    assert M.fit_cox([1.0, 2.0], [0, 0], [0.0, 1.0]).status == "ohne_ereignis"


def test_gleichstand_aller_zeiten_ist_ein_einziges_ereignis_mit_vier_abholungen():
    # alle vier zur selben Zeit: eine Risikomenge mit allen vier, vier Abholungen; die Log-Likelihood ist β·Σx − 4·log Σ e^{βx}, maximal bei β = 0 (Σx/4 = Mittel)
    t, e, x = [3.0, 3.0, 3.0, 3.0], [1, 1, 1, 1], [1.0, 0.0, 1.0, 0.0]
    fit = M.fit_cox(t, e, x)
    assert fit.status in ("ok", "singulaer")
    if fit.ok:
        assert fit.beta == pytest.approx(0.0, abs=1e-8)
    u, i = M.score_info(0.0, t, e, x)
    assert u == pytest.approx(0.0, abs=1e-14) and i == pytest.approx(4 * 0.25, abs=1e-14)


def test_alle_zensiert_und_konstantes_merkmal_geben_status_und_meldung():
    for fit in (M.fit_cox([1.0, 2.0, 3.0], [0, 0, 0], [0.0, 1.0, 2.0]), M.fit_cox([1.0, 2.0, 3.0], [1, 1, 1], [0.7, 0.7, 0.7])):
        assert not fit.ok and math.isnan(fit.beta) and math.isnan(fit.se) and fit.message


def test_ungueltige_zeiten_werfen_survival_error_und_die_simulation_liefert_nie_solche():
    with pytest.raises(K.SurvivalError):
        M.fit_cox([0.0, 1.0, 2.0], [1, 1, 1], [0.0, 1.0, 2.0])
    with pytest.raises(K.SurvivalError):
        M.fit_cox([float("nan"), 1.0], [1, 1], [0.0, 1.0])
    for w in (S.World(n=C.N_MIN, window=4, shape=C.SHAPE_MAX, beta=C.BETA_MAX), S.World(n=C.N_MAX, window=1000, shape=C.SHAPE_MIN, beta=C.BETA_MAX)):
        sm = S.simulate(w, 0)
        assert np.all(np.isfinite(sm.time)) and np.all(sm.time > 0)


def test_kleinste_stichprobe_hat_laeufe_ohne_schaetzer_mit_grund():
    s = M.summarize(M.run_study(S.World(n=C.N_MIN), 60), S.World(n=C.N_MIN))
    assert s.n_no_fit > 0 and s.n_ok + s.n_no_fit == s.reps
    assert all(k in M.STATUS_TEXT for k, _ in s.status_counts)
    assert "keinen endlichen Cox-Schätzer" in ev.failure_text(s)


def test_studie_ohne_einen_einzigen_schaetzer_bleibt_definiert():
    study = M.Study(beta=np.full(3, np.nan), se=np.full(3, np.nan), ols_all=np.full(3, np.nan), ols_events=np.full(3, np.nan), censored=np.ones(3), statuses=("ohne_ereignis",) * 3)
    s = M.summarize(study, S.World())
    assert s.n_ok == 0 and math.isnan(s.cox_mean) and s.coverage_hits == 0
    assert ev.verdict_study(s, S.World())[0] == "warning" and "Ohne Schätzer" in ev.coverage_text(s)


def test_beta_null_hat_wahrheit_null_ohne_division_durch_null():
    w = S.World(beta=0.0)
    s = M.summarize(M.run_study(w, 20), w)
    kind, text = ev.verdict_study(s, w)
    assert kind == "info" and "Ohne Wirkung" in text


def test_experiment_drei_ohne_vergleichsbereich_und_ohne_schaetzer():
    a = ev.analyse_curves(S.World(n=C.N_MIN))
    assert a.comparison is None
    kind, text = ev.verdict_curves(a)
    if a.fit.ok:
        assert kind == "info" and "Kein Vergleichsbereich" in text
    else:
        assert kind == "warning" and "Kein Schätzer" in text
