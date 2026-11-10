"""Orakel-Tests der Cox-Anpassung: ein anderer Rechenweg als der der Demo (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2).

Der Prüfling rechnet über Newton-Schritte mit gruppierten Risikomengen je verschiedener Ereigniszeit (`cxr_core.cox_fit`, `cxr_model.partial_loglik`).
Das Orakel hier rechnet in reinem Python je Container mit Ereignis (nicht je verschiedener Zeit) und sucht den Schätzer ohne Newton:
  * `scipy.optimize.brentq` auf der Nullstelle des Scores (Bisektion/Sekante, nur Funktionswerte),
  * `scipy.optimize.minimize_scalar` auf der negativen Log-Likelihood (Brent, ohne Ableitung),
  * Score und Information als numerische Differenzen der Log-Likelihood,
  * Trennung als fehlender Vorzeichenwechsel des Scores.
Zufall nur zum Erzeugen kleiner Instanzen (feste Seeds); jede Instanz mit Gleichständen durch Runden der Zeiten und mit Zensierung.
"""

import math

import numpy as np
import pytest

scipy_optimize = pytest.importorskip("scipy.optimize")

import cxr_core as K  # noqa: E402
import cxr_model as M  # noqa: E402
import cxr_scenario as S  # noqa: E402


def ref_terms(beta, t, e, x):
    """Je Container mit Ereignis: (x_i, log Σ_{j: t_j ≥ t_i} e^{β x_j}, m_i = gewichtetes Mittel der Risikomenge). Reine Schleifen."""
    out = []
    for i in range(len(t)):
        if not e[i]:
            continue
        risk = [j for j in range(len(t)) if t[j] >= t[i]]
        top = max(beta * x[j] for j in risk)
        w = [math.exp(beta * x[j] - top) for j in risk]
        out.append((x[i], top + math.log(sum(w)), sum(wj * x[j] for wj, j in zip(w, risk)) / sum(w)))
    return out


def ref_loglik(beta, t, e, x):
    return sum(beta * xi - lse for xi, lse, _ in ref_terms(beta, t, e, x))


def ref_score(beta, t, e, x):
    return sum(xi - m for xi, _, m in ref_terms(beta, t, e, x))


def ref_root(t, e, x, bound=40.0):
    """Nullstelle des Scores über Vorzeichenwechsel auf [−bound, bound] (brentq); None, wenn es keinen Wechsel gibt (Trennung oder keine Information)."""
    lo, hi = ref_score(-bound, t, e, x), ref_score(bound, t, e, x)
    if lo * hi >= 0.0:
        return None
    return scipy_optimize.brentq(lambda b: ref_score(b, t, e, x), -bound, bound, xtol=1e-13, rtol=1e-13)


def instance(seed, n=None):
    """Kleine Instanz: Merkmale aus {0, 1, 2} oder normal, Zeiten gerundet (Gleichstände), etwa ein Drittel zensiert."""
    rng = K.SplitMix64(seed)
    n = n or 6 + seed % 10
    x = rng.normal(n) if seed % 2 else np.floor(rng.uniform(n) * 3)
    t = np.maximum(np.round(rng.exponential(n) * 3 * np.exp(-0.5 * x), 0), 1.0)   # ganzzahlige Zeiten ab 1: viele Gleichstände
    e = rng.uniform(n) > 0.33
    return t, e, x


SEEDS = range(60)


def test_schaetzer_gleicht_der_nullstelle_des_scores_ohne_newton():
    checked = 0
    for seed in SEEDS:
        t, e, x = instance(seed)
        if not e.any():
            continue
        fit = M.fit_cox(t, e, x)
        root = ref_root(t, e, x)
        if fit.ok:
            assert root is not None, seed
            assert fit.beta == pytest.approx(root, abs=1e-7), seed
            checked += 1
    assert checked >= 25   # genug Instanzen mit endlichem Schätzer, damit der Vergleich etwas prüft


def test_schaetzer_gleicht_der_direkten_minimierung():
    checked = 0
    for seed in SEEDS:
        t, e, x = instance(seed)
        fit = M.fit_cox(t, e, x)
        if not fit.ok or abs(fit.beta) > 8:
            continue
        res = scipy_optimize.minimize_scalar(lambda b: -ref_loglik(b, t, e, x), bounds=(-10.0, 10.0), method="bounded", options={"xatol": 1e-10})
        assert fit.beta == pytest.approx(res.x, abs=1e-5), seed
        assert fit.loglik == pytest.approx(-res.fun, abs=1e-9), seed
        checked += 1
    assert checked >= 20


def test_handbeispiel_schaetzer_gegen_orakel(hand):
    t, e, x = hand
    root = ref_root(t, e, x)
    assert root == pytest.approx(0.9406136, abs=1e-6)
    assert M.fit_cox(t, e, x).beta == pytest.approx(root, abs=1e-8)
    assert ref_score(0.0, t, e, x) == pytest.approx(2 / 3, abs=1e-14)


def test_partielle_loglik_der_demo_gleicht_dem_orakel_bei_beliebigem_beta():
    for seed in SEEDS:
        t, e, x = instance(seed)
        for beta in (-1.3, 0.0, 0.4, 2.0):
            if e.any():
                assert M.partial_loglik(beta, t, e, x) == pytest.approx(ref_loglik(beta, t, e, x), abs=1e-10), (seed, beta)


def test_score_und_information_der_demo_gleichen_numerischen_differenzen_der_loglik():
    for seed in SEEDS:
        t, e, x = instance(seed)
        if not e.any():
            continue
        for beta in (-0.5, 0.3, 1.1):
            h = 1e-6
            num_score = (ref_loglik(beta + h, t, e, x) - ref_loglik(beta - h, t, e, x)) / (2 * h)
            h2 = 1e-3
            num_info = -(ref_loglik(beta + h2, t, e, x) - 2 * ref_loglik(beta, t, e, x) + ref_loglik(beta - h2, t, e, x)) / h2**2
            u, i = M.score_info(beta, t, e, x)
            assert u == pytest.approx(num_score, abs=1e-6), (seed, beta)
            assert i == pytest.approx(num_info, rel=1e-4, abs=1e-6), (seed, beta)
            assert u == pytest.approx(ref_score(beta, t, e, x), abs=1e-10)


def test_standardfehler_ist_eins_durch_wurzel_der_numerischen_information():
    checked = 0
    for seed in SEEDS:
        t, e, x = instance(seed, n=12 + seed % 8)
        fit = M.fit_cox(t, e, x)
        if not fit.ok:
            continue
        h = 1e-3
        info = -(ref_loglik(fit.beta + h, t, e, x) - 2 * ref_loglik(fit.beta, t, e, x) + ref_loglik(fit.beta - h, t, e, x)) / h**2
        assert fit.se == pytest.approx(1 / math.sqrt(info), rel=1e-4), seed
        checked += 1
    assert checked >= 20


def test_trennung_ist_genau_der_fehlende_vorzeichenwechsel_des_scores():
    seen = {"trennung": 0, "ok": 0}
    for seed in range(200):
        t, e, x = instance(seed, n=4 + seed % 5)     # kleine Instanzen: Trennung kommt vor
        if not e.any():
            continue
        fit = M.fit_cox(t, e, x)
        root = ref_root(t, e, x)
        if fit.status == "trennung":
            assert root is None, seed
            seen["trennung"] += 1
        elif fit.status == "singulaer":
            assert root is None, seed          # keine Information: der Score ist überall 0
        else:
            assert fit.ok and root is not None and fit.beta == pytest.approx(root, abs=1e-6), seed
            seen["ok"] += 1
    assert seen["trennung"] >= 3 and seen["ok"] >= 30   # beide Zweige kommen wirklich vor


def test_breslow_grundrate_gegen_summe_je_container_mit_ereignis():
    for seed in range(30):
        t, e, x = instance(seed, n=12)
        if not e.any():
            continue
        beta = 0.45
        ts, h0 = M.breslow_baseline(t, e, x, beta)
        for tk, hk in zip(ts, h0):
            ref = sum(1.0 / sum(math.exp(beta * x[j]) for j in range(len(t)) if t[j] >= t[i]) for i in range(len(t)) if e[i] and t[i] <= tk)
            assert hk == pytest.approx(ref, abs=1e-12), (seed, tk)


def test_simulation_mit_bekannter_wahrheit_als_band():
    """40 Läufe der Welt (n = 300, Fenster 12 Tage, β = 0,7, Form 1,5): Cox trifft β, die Regression auf die Zeit liegt bei weniger als der Hälfte von −β/k.
    Bänder aus der Messung dieser Demo (README, tools/measure_cox.py: Cox 0,70 ± 0,01 und Regression −0,21 bei 200 Läufen), hier großzügig für 40 Läufe."""
    study = M.run_study(S.World(), 40)
    assert np.isfinite(study.beta).all()
    assert abs(float(study.beta.mean()) - 0.7) < 0.06
    assert -0.27 < float(study.ols_all.mean()) < -0.15
    assert abs(float(study.ols_all.mean())) < 0.7 * abs(-0.7 / 1.5)        # weniger als 70 % der Wahrheit −β/k (in Betrag)


def test_ein_einziges_ereignis_ist_trennung_obwohl_der_kern_konvergenz_meldet():
    """Fund des Orakels (Instanz 197 der Trennungsprobe): ein einziges Ereignis, das Merkmal des Abgeholten ist das größte seiner Risikomenge.
    Eine frühere Fassung des Kerns meldete hier `ok = True` mit β̂ = 16,4 und Standardfehler 9,5·10^7, weil die Log-Likelihood in Gleitkomma schon bei 0 liegt;
    die aktuelle Kopie (`cxr_core.cox_fit`) liefert wegen einer Plausibilitätsgrenze `ok = False`. Der Wrapper `cxr_model.fit_cox` erkennt die Trennung selbst und gibt keinen Schätzer aus."""
    t, e, x = instance(197, n=4 + 197 % 5)
    assert int(e.sum()) == 1 and ref_root(t, e, x) is None
    fit = M.fit_cox(t, e, x)
    assert fit.status == "trennung" and not fit.ok and math.isnan(fit.beta)
    assert M.fit_cox([1.0, 2.0, 3.0], [1, 0, 0], [1.0, 0.0, -1.0]).status == "trennung"   # Handbeispiel: ein Ereignis, größtes Merkmal zuerst
