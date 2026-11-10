"""Einheiten des Modells an der Mini-Instanz (Handbeispiel, Brüche von Hand nachgerechnet) und am Gleichstands-Beispiel mit geschlossener Lösung."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import cxr_constants as C
import cxr_core as K
import cxr_model as M
import cxr_scenario as S


def test_partielle_loglik_bei_null_ist_minus_log_vierundzwanzig(hand):
    t, e, x = hand
    # bei β = 0 ist jede Wahrscheinlichkeit 1/|Risikomenge|: −(log 4 + log 3 + log 2 + log 1) = −log 24
    assert M.partial_loglik(0.0, t, e, x) == pytest.approx(-math.log(24), abs=1e-12)


def test_score_und_information_bei_null_sind_die_handbrueche(hand):
    t, e, x = hand
    u, i = M.score_info(0.0, t, e, x)
    assert u == pytest.approx(float(F(2, 3)), abs=1e-14)       # 1/2 − 1/3 + 1/2 + 0
    assert i == pytest.approx(float(F(13, 18)), abs=1e-14)     # 1/4 + 2/9 + 1/4 + 0


def test_tabelle_der_risikomengen_in_bruechen(hand):
    t, e, x = hand
    rows = M.risk_rows_exact(t, e, x)
    assert [r["risk_x"] for r in rows] == [[1, 0, 1, 0], [0, 1, 0], [1, 0], [0]]
    assert [r["m"] for r in rows] == [F(1, 2), F(1, 3), F(1, 2), F(0)]
    assert [r["score"] for r in rows] == [F(1, 2), F(-1, 3), F(1, 2), F(0)]
    assert [r["info"] for r in rows] == [F(1, 4), F(2, 9), F(1, 4), F(0)]
    assert sum(r["score"] for r in rows) == F(2, 3) and sum(r["info"] for r in rows) == F(13, 18)


def test_tabelle_in_gleitkomma_stimmt_bei_null_mit_den_bruechen_ueberein(hand):
    t, e, x = hand
    for exact, flt in zip(M.risk_rows_exact(t, e, x), M.risk_rows(0.0, t, e, x)):
        assert flt["m"] == pytest.approx(float(exact["m"]), abs=1e-15)
        assert flt["score"] == pytest.approx(float(exact["score"]), abs=1e-15)
        assert flt["info"] == pytest.approx(float(exact["info"]), abs=1e-15)


def test_exakte_tabelle_verlangt_ganzzahlige_merkmale():
    with pytest.raises(K.SurvivalError):
        M.risk_rows_exact([1.0, 2.0], [1, 1], [0.5, 1.0])


def test_newton_pfad_erster_schritt_ist_zwoelf_dreizehntel(hand):
    t, e, x = hand
    path = M.newton_path(t, e, x)
    assert path[0]["beta"] == 0.0
    assert path[1]["beta"] == pytest.approx(12 / 13, abs=1e-14)    # (2/3) / (13/18) = 12/13 = 0,923
    assert path[-1]["beta"] == pytest.approx(0.9406, abs=5e-5)
    assert all(b["loglik"] >= a["loglik"] - 1e-12 for a, b in zip(path, path[1:]))   # hier steigt die Log-Likelihood bei jedem Schritt
    assert path[-1]["beta"] == pytest.approx(K.cox_fit(t, e, x)["beta"][0], abs=1e-8)


def test_newton_pfad_haelt_bei_information_null_an():
    path = M.newton_path([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], [1.0, 1.0, 1.0, 1.0])
    assert len(path) == 1


def test_newton_pfad_bei_trennung_waechst_weiter_und_endet_am_schrittlimit():
    path = M.newton_path([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], [1.0, 1.0, 0.0, 0.0], max_steps=6)
    assert len(path) == 7 and all(b["beta"] > a["beta"] for a, b in zip(path, path[1:]))


def test_gleichstands_beispiel_hat_die_geschlossene_loesung_ln_zwei():
    # Zeiten 1, 1, 2, 3, Merkmale 1, 0, 1, 0, alle Ereignisse. Breslow: Score 2 − 3 e^β/(e^β + 1) = 0 ergibt e^β = 2.
    t, e, x = [1, 1, 2, 3], [1, 1, 1, 1], [1, 0, 1, 0]
    fit = M.fit_cox(t, e, x)
    assert fit.ok and fit.beta == pytest.approx(math.log(2), abs=1e-8)
    assert fit.se == pytest.approx(1 / math.sqrt(2 / 3), abs=1e-8)   # Information 4/9 + 2/9 = 2/3
    u, i = M.score_info(math.log(2), t, e, x)
    assert u == pytest.approx(0.0, abs=1e-12) and i == pytest.approx(2 / 3, abs=1e-12)


@pytest.mark.parametrize("pattern,kind", [
    ("Handbeispiel: x = 1, 0, 1, 0", "keine"),
    ("Umgekehrt: x = 0, 1, 0, 1", "keine"),
    ("Perfekte Trennung: x = 1, 1, 0, 0", "trennung"),
    ("Konstantes Merkmal: x = 1, 1, 1, 1", "flach"),
])
def test_trennungserkennung_der_vier_muster(pattern, kind):
    assert M.separation_kind(S.HAND_TIMES, np.ones(4, bool), S.PATTERNS[pattern]) == kind


def test_trennung_auch_in_die_andere_richtung_und_mit_zensierung():
    assert M.separation_kind([1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 1, 1]) == "trennung"      # immer der kleinste zuerst
    assert M.separation_kind([1, 2, 3, 4], [1, 0, 1, 1], [1, 0, 0, 1]) == "keine"           # der Zensierte bleibt in der Risikomenge und stellt sich quer
    assert M.separation_kind([1, 2], [0, 0], [0, 1]) == "keine"                              # ohne Ereignis


@pytest.mark.parametrize("pattern,status", [
    ("Handbeispiel: x = 1, 0, 1, 0", "ok"),
    ("Umgekehrt: x = 0, 1, 0, 1", "ok"),
    ("Perfekte Trennung: x = 1, 1, 0, 0", "trennung"),
    ("Konstantes Merkmal: x = 1, 1, 1, 1", "singulaer"),
])
def test_status_der_anpassung_fuer_die_vier_muster(pattern, status):
    fit = M.fit_cox(S.HAND_TIMES, np.ones(4, bool), S.PATTERNS[pattern])
    assert fit.status == status and fit.ok == (status == "ok")
    if status != "ok":
        assert math.isnan(fit.beta) and math.isnan(fit.se) and fit.message   # keine stille Zahl, dafür eine Meldung
    else:
        assert fit.message == ""


def test_umgekehrtes_muster_hat_das_entgegengesetzte_vorzeichen():
    a = M.fit_cox(S.HAND_TIMES, np.ones(4, bool), S.PATTERNS["Handbeispiel: x = 1, 0, 1, 0"])
    b = M.fit_cox(S.HAND_TIMES, np.ones(4, bool), S.PATTERNS["Umgekehrt: x = 0, 1, 0, 1"])
    assert b.beta == pytest.approx(-a.beta, abs=1e-8)
    assert a.hazard_ratio == pytest.approx(2.5616, abs=5e-4) and b.hazard_ratio == pytest.approx(1 / 2.5616, abs=5e-4)


def test_ohne_ereignis_gibt_status_statt_zahl():
    fit = M.fit_cox([1.0, 2.0, 3.0], [0, 0, 0], [0.0, 1.0, 2.0])
    assert fit.status == "ohne_ereignis" and not fit.ok and math.isnan(fit.beta)


def test_ols_steigung_von_hand():
    assert M.ols_slope([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0, abs=1e-14)
    assert M.ols_slope([1.0, 5.0, 3.0, 9.0], [0.0, 1.0, 2.0, 3.0]) == pytest.approx(2.2, abs=1e-14)   # Σ(x−1,5)(y−4,5)/Σ(x−1,5)² = 11/5
    assert M.ols_slope([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], mask=[True, True, True, False]) == pytest.approx(1.0, abs=1e-14)


def test_ols_ohne_steigung_gibt_nan():
    assert math.isnan(M.ols_slope([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]))      # konstantes Merkmal
    assert math.isnan(M.ols_slope([1.0, 2.0], [0.0, 1.0]))                 # weniger als drei Zeilen
    assert math.isnan(M.ols_slope([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], mask=[True, True, False, False]))


def test_breslow_grundrate_bei_beta_null_von_hand(hand):
    t, e, x = hand
    ts, h0 = M.breslow_baseline(t, e, x, 0.0)
    # Zuwächse 1/4, 1/3, 1/2, 1 (Risikomengen der Größe 4, 3, 2, 1)
    assert ts.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert h0.tolist() == pytest.approx([1 / 4, 7 / 12, 13 / 12, 25 / 12], abs=1e-14)


def test_breslow_grundrate_mit_zensierung_und_gleichstand():
    # Zeiten 1 (Ereignis), 1 (Ereignis), 2 (zensiert), 3 (Ereignis), β = 0: Zuwächse 2/4 bei t = 1 und 1/1 bei t = 3
    ts, h0 = M.breslow_baseline([1, 1, 2, 3], [1, 1, 0, 1], [0, 0, 0, 0], 0.0)
    assert ts.tolist() == [1.0, 3.0] and h0.tolist() == pytest.approx([0.5, 1.5], abs=1e-14)


def test_grundrate_als_treppe_und_vorhergesagte_kurve(hand):
    t, e, x = hand
    ts, h0 = M.breslow_baseline(t, e, x, 0.0)
    assert M.baseline_at(ts, h0, [0.5, 1.0, 1.5, 2.0, 10.0]).tolist() == pytest.approx([0.0, 1 / 4, 1 / 4, 7 / 12, 25 / 12], abs=1e-14)
    s = M.predicted_group_survival(ts, h0, 0.0, [1.0, 0.0], [0.5, 2.0])
    assert s.tolist() == pytest.approx([1.0, math.exp(-7 / 12)], abs=1e-14)            # bei β = 0 hängt die Kurve nicht vom Merkmal ab
    s1 = M.predicted_group_survival(ts, h0, math.log(2), [1.0], [2.0])
    assert s1[0] == pytest.approx(math.exp(-7 / 12 * 2), abs=1e-14)                    # e^{β x} = 2 verdoppelt die kumulierte Rate


def test_gruppen_nach_terzilen_des_merkmals():
    assert M.group_of([-2.0, -0.4308, -0.43, 0.0, 0.4307, 0.4308, 3.0]).tolist() == [0, 0, 1, 1, 2, 2, 2]


def test_vergleichsbereich_endet_wo_eine_gruppe_unter_zehn_im_risiko_faellt():
    sm = S.simulate(S.World(n=300, window=12), 0)
    groups = M.group_of(sm.x)
    tau = M.comparison_limit(sm, groups)
    assert 0 < tau <= float(sm.time[sm.event].max())
    for g in range(3):
        m = groups == g
        assert np.sum(sm.time[m] >= tau) >= C.MIN_AT_RISK   # zur Zeit tau sind in jeder Gruppe noch mindestens zehn im Risiko
    small = S.simulate(S.World(n=10, window=12), 0)
    assert math.isnan(M.comparison_limit(small, M.group_of(small.x)))


def test_zusammenfassung_von_hand_gesetzten_laeufen():
    study = M.Study(beta=np.array([0.5, 0.7, np.nan, 0.9]), se=np.array([0.1, 0.1, np.nan, 0.1]), ols_all=np.array([-0.2, -0.2, -0.2, -0.2]),
                    ols_events=np.array([-0.1, -0.3, np.nan, -0.2]), censored=np.array([0.5, 0.5, 0.5, 0.5]), statuses=("ok", "ok", "trennung", "ok"))
    s = M.summarize(study, S.World(beta=0.7, shape=1.4))
    assert (s.reps, s.n_ok, s.n_no_fit) == (4, 3, 1) and s.status_counts == (("ok", 3), ("trennung", 1))
    assert s.cox_mean == pytest.approx(0.7) and s.cox_sd == pytest.approx(0.2)
    assert s.cox_se_of_mean == pytest.approx(0.2 / math.sqrt(3))
    assert s.coverage_hits == 1             # |0,5 − 0,7| = 0,2 > 1,96 · 0,1 und |0,9 − 0,7| = 0,2 > 0,196; nur 0,7 liegt im Intervall
    assert s.ols_events_mean == pytest.approx(-0.2) and s.truth_ols == pytest.approx(-0.5)
    assert s.censored == pytest.approx(0.5) and s.mean_se == pytest.approx(0.1)


# ------------------------------------------------------------------------------------------------ Szenario


def test_wahre_zeit_und_ueberlebensfunktion_in_geschlossener_form():
    # E = 1, x = 0: T = λ; S(λ | 0) = exp(−1). Mit x = 1 und β = ln 2 verdoppelt sich die Rate: T halbiert sich bei k = 1.
    assert S.true_time(0.0, 1.0, 0.7, 1.5) == pytest.approx(C.WEIBULL_SCALE)
    assert S.true_time(1.0, 1.0, math.log(2), 1.0) == pytest.approx(C.WEIBULL_SCALE / 2)
    assert float(S.true_survival(C.WEIBULL_SCALE, 0.0, 0.7, 1.5)) == pytest.approx(math.exp(-1))
    assert float(S.true_survival(C.WEIBULL_SCALE, 1.0, math.log(2), 1.5)) == pytest.approx(math.exp(-2))       # e^{β x} = 2 verdoppelt die kumulierte Rate


def test_simulation_ist_bei_gleichem_seed_gleich_und_je_lauf_verschieden():
    w = S.World(n=50, seed=7)
    a, b, c = S.simulate(w, 0), S.simulate(w, 0), S.simulate(w, 1)
    assert np.array_equal(a.time, b.time) and np.array_equal(a.event, b.event) and np.array_equal(a.x, b.x)
    assert not np.array_equal(a.time, c.time)
    assert S.run_seed(7, 0) != S.run_seed(7, 1) and S.run_seed(7, 1) != S.run_seed(8, 1)


def test_simulation_beobachtet_das_minimum_aus_wahrer_zeit_und_zensierzeit():
    sm = S.simulate(S.World(n=500, window=12), 0)
    assert np.all(sm.time > 0) and np.all(sm.time <= sm.latent + 1e-9)
    assert np.all(sm.time[sm.event] == pytest.approx(sm.latent[sm.event]))        # Ereignis: beobachtet wird die wahre Zeit
    assert np.all(sm.time[~sm.event] < sm.latent[~sm.event]) and np.all(sm.time[~sm.event] <= 12.0)   # zensiert: die Zensierzeit liegt im Fenster und vor der Abholung


def test_ohne_zensierung_ist_praktisch_jeder_container_abgeholt():
    sm = S.simulate(S.World(n=500, window=1000), 0)
    assert sm.event.mean() > 0.95
