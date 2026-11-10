"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für die Cox-Anpassung gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 40 feste Instanzen (Normal-Merkmal, 55 % zensiert, n = 60 bis 120), Zufall nur über SplitMix64 mit festen Seeds.
Gelten die Eigenschaften nur unter Voraussetzungen, stehen sie im Docstring."""

import numpy as np
import pytest

import cxr_model as M
import cxr_scenario as S

RUNS = range(25)


def data(run, n=80, beta=0.7, window=12.0):
    sm = S.simulate(S.World(n=n, window=window, beta=beta, seed=11), run)
    return sm.time, sm.event, sm.x


def beta_of(t, e, x):
    fit = M.fit_cox(t, e, x)
    assert fit.ok
    return fit.beta


def test_skalierung_des_merkmals_skaliert_beta_mit_eins_durch_c():
    for run in RUNS:
        t, e, x = data(run)
        b = beta_of(t, e, x)
        for c in (2.0, 0.5, -1.5):
            assert beta_of(t, e, c * x) == pytest.approx(b / c, abs=1e-7), (run, c)


def test_verschiebung_des_merkmals_aendert_nichts():
    """Gilt, weil sich die Verschiebung in jedem Gewichtsquotienten der Risikomenge herauskürzt (ein Merkmal, Breslow)."""
    for run in RUNS:
        t, e, x = data(run)
        assert beta_of(t, e, x + 3.7) == pytest.approx(beta_of(t, e, x), abs=1e-7), run


def test_zeitverschiebung_und_streng_wachsende_zeittransformation_aendern_nichts():
    """Die Schätzung hängt nur von der Reihenfolge der Zeiten (und vom Ereignis-Indikator) ab."""
    for run in RUNS:
        t, e, x = data(run)
        b = beta_of(t, e, x)
        assert beta_of(t + 10.0, e, x) == pytest.approx(b, abs=1e-9), run
        assert beta_of(np.log1p(t), e, x) == pytest.approx(b, abs=1e-9), run
        assert beta_of(3.0 * t**2, e, x) == pytest.approx(b, abs=1e-9), run


def test_umsortieren_der_container_aendert_nichts():
    rng = np.random.default_rng(0)
    for run in RUNS:
        t, e, x = data(run)
        p = rng.permutation(len(t))
        assert beta_of(t[p], e[p], x[p]) == pytest.approx(beta_of(t, e, x), abs=1e-9), run


def test_vertauschte_gruppen_wechseln_das_vorzeichen():
    for run in RUNS:
        t, e, x = data(run)
        binary = (x > 0).astype(float)
        assert beta_of(t, e, 1.0 - binary) == pytest.approx(-beta_of(t, e, binary), abs=1e-7), run


def test_ohne_wirkung_streut_beta_um_null():
    """Zeiten (mit Ereignis-Indikator) zwischen den Containern permutiert: das Merkmal hat dann keine Wirkung; Mittel über 200 Permutationen nahe 0,
    gemessen am Standardfehler des Mittels (Bandbreite vier Standardfehler)."""
    t, e, x = data(0, n=100)
    rng = np.random.default_rng(1)
    betas = []
    for _ in range(200):
        p = rng.permutation(len(t))
        betas.append(beta_of(t[p], e[p], x))
    betas = np.array(betas)
    assert abs(betas.mean()) < 4 * betas.std(ddof=1) / np.sqrt(len(betas))


def test_mehr_container_geben_einen_kleineren_standardfehler_und_naehere_schaetzer():
    """Mittel über 40 Läufe: Standardfehler und mittlerer Abstand zur Wahrheit 0,7 sind bei n = 400 kleiner als bei n = 40."""
    def stats(n):
        fits = [M.fit_cox(*(lambda s: (s.time, s.event, s.x))(S.simulate(S.World(n=n, seed=11), r))) for r in range(40)]
        ok = [f for f in fits if f.ok]
        return float(np.mean([f.se for f in ok])), float(np.mean([abs(f.beta - 0.7) for f in ok]))

    se_small, err_small = stats(40)
    se_big, err_big = stats(400)
    assert se_big < se_small and err_big < err_small


def test_staerkere_wirkung_gibt_groesseres_beta():
    """Gleiche Zufallszahlen, nur β der Welt wächst: das geschätzte β wächst mit (Mittel über 25 Läufe bei 0, 0,7, 1,4)."""
    means = [np.mean([beta_of(*data(r, beta=b)) for r in RUNS]) for b in (0.0, 0.7, 1.4)]
    assert means[0] < means[1] < means[2]


def test_gleicher_seed_gleiche_schaetzung_im_zweiten_aufruf():
    a = M.run_study(S.World(n=60, seed=3), 10)
    b = M.run_study(S.World(n=60, seed=3), 10)
    assert np.array_equal(a.beta, b.beta, equal_nan=True) and np.array_equal(a.ols_all, b.ols_all, equal_nan=True)


def test_container_die_vor_dem_ersten_ereignis_zensiert_werden_aendern_nichts():
    """Wer vor der ersten Ereigniszeit zensiert wird, steht in keiner Risikomenge eines Ereignisses (Voraussetzung: seine Zeit ist kleiner als jede Ereigniszeit)."""
    for run in RUNS:
        t, e, x = data(run)
        first = float(t[e].min())
        t2 = np.concatenate([t, first * np.array([0.1, 0.2, 0.3])])
        e2 = np.concatenate([e, [False, False, False]])
        x2 = np.concatenate([x, [5.0, -5.0, 0.3]])
        assert beta_of(t2, e2, x2) == pytest.approx(beta_of(t, e, x), abs=1e-9), run
