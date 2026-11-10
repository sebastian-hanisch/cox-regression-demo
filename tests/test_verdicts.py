"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math

import cxr_evaluation as ev
import cxr_model as M
import cxr_scenario as S

PATTERN_TRENNUNG = "Perfekte Trennung: x = 1, 1, 0, 0"
PATTERN_KONSTANT = "Konstantes Merkmal: x = 1, 1, 1, 1"


def summary(**kw) -> M.Summary:
    base = dict(reps=200, n_ok=200, n_no_fit=0, status_counts=(("ok", 200),), censored=0.55, cox_mean=0.70, cox_se_of_mean=0.01, cox_sd=0.10, mean_se=0.09,
                coverage_hits=186, ols_all_mean=-0.20, ols_all_se_of_mean=0.004, ols_events_mean=-0.21, ols_events_se_of_mean=0.006, truth_cox=0.7, truth_ols=-0.5)
    base.update(kw)
    return M.Summary(**base)


def test_zahlenformat_mit_dezimalkomma_und_echtem_minus():
    assert ev.num(0.9406136, 4) == "0,9406" and ev.num(-0.2063, 2) == "−0,21" and ev.pct(0.456) == "46 %" and ev.pct(0.93, 1) == "93,0 %"


def test_bruchdarstellung():
    from fractions import Fraction as F

    assert ev.frac(F(13, 18)) == "13/18" and ev.frac(F(-1, 3)) == "−1/3" and ev.frac(F(0)) == "0" and ev.frac(F(4, 2)) == "2"


def test_verdikt_hand_nennt_beta_hazard_verhaeltnis_und_standardfehler():
    kind, text = ev.verdict_hand(ev.analyse_hand("Handbeispiel: x = 1, 0, 1, 0", 0.0))
    assert kind == "success"
    assert "β̂ = 0,9406" in text and "2,562" in text and "Standardfehler ist 1,24" in text and "nur zum Nachrechnen" in text


def test_verdikt_hand_bei_trennung_und_konstantem_merkmal_ist_eine_warnung_ohne_zahl():
    kind, text = ev.verdict_hand(ev.analyse_hand(PATTERN_TRENNUNG, 0.0))
    assert kind == "warning" and "Kein Schätzer" in text and "Perfekte Trennung" in text and "β̂ =" not in text
    kind, text = ev.verdict_hand(ev.analyse_hand(PATTERN_KONSTANT, 0.0))
    assert kind == "warning" and "Information über β ist 0" in text and "β̂ =" not in text


def test_analyse_hand_probe_werte():
    a = ev.analyse_hand("Handbeispiel: x = 1, 0, 1, 0", 0.0)
    assert abs(a.probe_score - 2 / 3) < 1e-14
    assert abs(a.probe_info - 13 / 18) < 1e-14 and len(a.exact_rows) == 4 and len(a.probe_rows) == 4


def test_verdikt_studie_regression_unter_neun_zehnteln_ist_erfolg_mit_beiden_anteilen():
    w = S.World(n=300, window=12)
    kind, text = ev.verdict_study(summary(), w)                         # Regression −0,20 von −0,50: 40 %; Cox 0,70 von 0,7: 100 %
    assert kind == "success"
    assert "Über 200 Läufe mit n = 300 Containern und Beobachtungsfenster 12 Tage (55 % zensiert)" in text
    assert "Cox β̂ im Mittel 0,70 (Wahrheit 0,70)" in text and "Steigung −0,20 über alle Container und −0,21 nur über die Ereignisse" in text
    assert "(Wahrheit −β/k = −0,50)" in text and "**40 %** der Wahrheit, Cox **100 %**" in text


def test_verdikt_studie_schwelle_bei_neun_zehnteln():
    w = S.World()
    kind, text = ev.verdict_study(summary(ols_all_mean=-0.4499), w)     # 89,98 % der Wahrheit −0,5: noch verzerrt
    assert kind == "success" and "**90 %** der Wahrheit" in text
    kind, text = ev.verdict_study(summary(ols_all_mean=-0.4501), w)     # 90,02 %: nicht mehr
    assert kind == "info" and "bei 90 % der Wahrheit" in text


def test_verdikt_studie_ohne_zensierung_ist_info_ohne_verzerrungsaussage():
    kind, text = ev.verdict_study(summary(censored=0.008, ols_all_mean=-0.48, ols_events_mean=-0.49), S.World(window=1000))
    assert kind == "info" and "96 % der Wahrheit" in text and "nicht deutlich zur Null verzerrt" in text


def test_verdikt_studie_ohne_wirkung_und_ohne_schaetzer():
    kind, text = ev.verdict_study(summary(truth_cox=0.0, truth_ols=0.0, cox_mean=0.006, ols_all_mean=0.005, ols_events_mean=0.007), S.World(beta=0.0))
    assert kind == "info" and "Ohne Wirkung" in text and "0,01" in text
    kind, text = ev.verdict_study(summary(n_ok=0, n_no_fit=200), S.World())
    assert kind == "warning" and "In keinem der 200 Läufe" in text


def test_ueberdeckungstext_zaehlt_k_von_m_und_nennt_den_binomialfehler():
    text = ev.coverage_text(summary(coverage_hits=186))
    p = 186 / 200
    assert "**186 von 200 Läufen** (93,0 %" in text
    assert f"{ev.num(100 * math.sqrt(p * (1 - p) / 200), 1)} Prozentpunkte" in text and "1,8 Prozentpunkte" in text
    assert "Mittlerer Standardfehler 0,090, tatsächliche Streuung der β̂ über die Läufe 0,100" in text


def test_ausfalltext_nennt_zahl_und_gruende():
    assert ev.failure_text(summary()) == ""
    text = ev.failure_text(summary(n_ok=191, n_no_fit=9, status_counts=(("ok", 191), ("trennung", 9))))
    assert "**9 von 200 Läufen**" in text and "perfekte Trennung: 9" in text and "ok:" not in text
    text = ev.failure_text(summary(n_ok=190, n_no_fit=10, status_counts=(("ohne_ereignis", 4), ("ok", 190), ("singulaer", 6))))
    assert "ohne Ereignis: 4" in text and "Information 0: 6" in text


def test_verdikt_kurven_nennt_tau_abstaende_und_vergleich_mit_der_wahrheit():
    cmp_ = M.CurveComparison(tau=6.34, gaps=(0.05, 0.081, 0.062), truth_gaps=(0.04, 0.07, 0.03), km_truth_gaps=(0.06, 0.11, 0.05))
    fit = M.CoxFit(0.7, 0.1, True, "ok", 5, -100.0)
    a = ev.CurveAnalysis(S.simulate(S.World(n=30), 0), fit, cmp_)
    kind, text = ev.verdict_curves(a)
    assert kind == "info" and "Bis Tag 6,3 " in text and "mindestens 10 Container" in text
    assert "höchstens **8,1 Prozentpunkte**" in text and "(je Gruppe 5,0, 8,1, 6,2)" in text
    assert "um höchstens 11,0 Prozentpunkte von der wahren Kurve ab, die Cox-Kurve um höchstens 7,0" in text


def test_verdikt_kurven_ohne_schaetzer_ist_eine_warnung():
    fit = M.CoxFit(float("nan"), float("nan"), False, "trennung", 0, float("nan"))
    kind, text = ev.verdict_curves(ev.CurveAnalysis(S.simulate(S.World(n=30), 0), fit, None))
    assert kind == "warning" and "Kein Schätzer, keine Cox-Kurven" in text and "Perfekte Trennung" in text
