"""Cox-Modell der Demo: erklärende Einheiten (partielle Log-Likelihood je Ereigniszeit, Newton-Pfad), Trennungserkennung,
Status statt stiller Zahlen, Breslow-Grundrate, Regression auf die Zeit und die Simulationsstudie über feste Seeds.

Der Kern (Newton mit Schrittweitenhalbierung, Konvergenzflag) ist `cxr_core.cox_fit` (Kopie). Hier steht, was die Demo zeigt:
Wo das Verfahren selbst vorgeführt wird, geht Nachvollziehbarkeit vor (DEMO-PLAYBOOK §1c), deshalb laufen die Handbeispiel-Einheiten als einfache
Schleifen über Ereigniszeiten. Jede Einheit hat eine eigene Funktion mit expliziten Ein- und Ausgaben."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import cxr_constants as C
import cxr_core as K
from cxr_scenario import Sample, World, simulate, true_survival

# Status der Anpassung: Wörter statt stiller Zahlen (jede Nicht-"ok"-Zeile bekommt in der App eine Meldung)
STATUS_TEXT = {
    "ok": "",
    "ohne_ereignis": "Es gibt kein einziges beobachtetes Ereignis (alle Container sind zensiert): ohne Ereignis gibt es keine Information über das Merkmal.",
    "singulaer": "Das Merkmal ist in jeder Risikomenge gleich (oder konstant): die Information über β ist 0, ein Schätzer ist nicht bestimmt.",
    "trennung": "Perfekte Trennung: In jeder Risikomenge wird immer der Container mit dem größten (oder immer mit dem kleinsten) Merkmalswert zuerst abgeholt. "
                "Die partielle Likelihood steigt dann für β gegen unendlich immer weiter; es gibt keinen endlichen Schätzer.",
    "keine_konvergenz": "Das Newton-Verfahren ist nicht konvergiert; das Ergebnis wird nicht angezeigt.",
}


# ------------------------------------------------------------------------------------------------ Einheiten des Handbeispiels


def _risk_set(time, t):
    """Risikomenge zur Zeit t: alle, die zu t noch da sind (Zeit >= t); bei gleicher Zeit zählt das Ereignis vor der Zensierung."""
    return np.asarray(time, dtype=float) >= t


def partial_loglik(beta: float, time, event, x) -> float:
    """Partielle Log-Likelihood (Breslow) über die verschiedenen Ereigniszeiten: Σ_k [ β·Σ_{i ∈ D_k} x_i − d_k · log Σ_{j ∈ R_k} exp(β·x_j) ]."""
    t, e, xv = np.asarray(time, float), np.asarray(event, bool), np.asarray(x, float)
    total = 0.0
    for tk in np.unique(t[e]):
        at_risk = _risk_set(t, tk)
        dk = (e & (t == tk))
        eta = beta * xv[at_risk]
        shift = float(eta.max())
        total += beta * float(xv[dk].sum()) - float(dk.sum()) * (shift + math.log(float(np.exp(eta - shift).sum())))
    return total


def score_info(beta: float, time, event, x) -> tuple[float, float]:
    """Score U(β) = Σ_k Σ_{i ∈ D_k}(x_i − m_k(β)) und Information I(β) = Σ_k d_k · Varianz von x in der Risikomenge R_k mit Gewichten exp(β·x)."""
    t, e, xv = np.asarray(time, float), np.asarray(event, bool), np.asarray(x, float)
    u = info = 0.0
    for tk in np.unique(t[e]):
        at_risk = _risk_set(t, tk)
        dk = (e & (t == tk))
        w = np.exp(beta * xv[at_risk] - beta * float(xv[at_risk].max()))
        m = float((w * xv[at_risk]).sum() / w.sum())
        v = float((w * xv[at_risk] ** 2).sum() / w.sum()) - m * m
        u += float((xv[dk] - m).sum())
        info += float(dk.sum()) * v
    return u, info


def risk_rows(beta: float, time, event, x) -> list[dict]:
    """Eine Zeile je Ereigniszeit für die Tabelle der Risikomengen: t, d, Merkmale der Risikomenge, gewichtetes Mittel m, Beitrag zum Score, Beitrag zur Information."""
    t, e, xv = np.asarray(time, float), np.asarray(event, bool), np.asarray(x, float)
    rows = []
    for tk in np.unique(t[e]):
        at_risk = _risk_set(t, tk)
        dk = (e & (t == tk))
        w = np.exp(beta * xv[at_risk] - beta * float(xv[at_risk].max()))
        m = float((w * xv[at_risk]).sum() / w.sum())
        v = float((w * xv[at_risk] ** 2).sum() / w.sum()) - m * m
        rows.append({"t": float(tk), "d": int(dk.sum()), "risk_x": xv[at_risk].tolist(), "m": m, "score": float((xv[dk] - m).sum()), "info": float(dk.sum()) * v})
    return rows


def risk_rows_exact(time, event, x) -> list[dict]:
    """Dieselbe Tabelle bei β = 0 in Brüchen (Merkmale müssen ganzzahlig sein): Mittel, Beitrag zum Score und zur Information exakt."""
    t, e = np.asarray(time, float), np.asarray(event, bool)
    xv = [int(v) for v in x]
    if any(float(a) != float(b) for a, b in zip(xv, x)):
        raise K.SurvivalError("Die exakte Tabelle braucht ganzzahlige Merkmale.")
    rows = []
    for tk in np.unique(t[e]):
        at_risk = [i for i in range(len(xv)) if t[i] >= tk]
        dk = [i for i in at_risk if e[i] and t[i] == tk]
        n = len(at_risk)
        m = Fraction(sum(xv[i] for i in at_risk), n)
        v = Fraction(sum(xv[i] ** 2 for i in at_risk), n) - m * m
        rows.append({"t": float(tk), "d": len(dk), "risk_x": [xv[i] for i in at_risk], "m": m, "score": sum(xv[i] - m for i in dk), "info": len(dk) * v})
    return rows


def newton_path(time, event, x, start: float = 0.0, max_steps: int = 12, tol: float = 1e-10) -> list[dict]:
    """Newton-Schritte β_{j+1} = β_j + U(β_j)/I(β_j) von `start` aus, ohne Schrittweitenhalbierung (so wie man es von Hand rechnet).
    Jede Zeile: β, Score, Information, Log-Likelihood. Das Verfahren hält an, wenn |Schritt| < tol, die Information 0 ist (kein Schritt möglich)
    oder die Schrittzahl erreicht ist (bei perfekter Trennung wächst β bei jedem Schritt weiter)."""
    path = []
    beta = float(start)
    for _ in range(max_steps + 1):
        u, i = score_info(beta, time, event, x)
        path.append({"beta": beta, "score": u, "info": i, "loglik": partial_loglik(beta, time, event, x)})
        if i <= 1e-12:
            break
        step = u / i
        if abs(step) < tol:
            break
        beta += step
    return path


# ------------------------------------------------------------------------------------------------ Trennung und Status


def separation_kind(time, event, x) -> str:
    """„trennung“, „flach“ oder „keine“ für ein Merkmal. Trennung: In jeder Risikomenge hat jeder Container mit Ereignis den größten (oder in jeder den
    kleinsten) Merkmalswert; die partielle Likelihood ist dann in β monoton und hat kein endliches Maximum. „flach“: beide Bedingungen zugleich
    (jede Risikomenge hat nur einen Merkmalswert: die Information ist 0). Ohne Ereignis „keine“."""
    t, e, xv = np.asarray(time, float), np.asarray(event, bool), np.asarray(x, float)
    ts = np.unique(t[e])
    if ts.size == 0:
        return "keine"
    all_max = all_min = True
    for tk in ts:
        at_risk = _risk_set(t, tk)
        xe = xv[e & (t == tk)]
        hi, lo = float(xv[at_risk].max()), float(xv[at_risk].min())
        all_max = all_max and bool(np.all(xe == hi))
        all_min = all_min and bool(np.all(xe == lo))
    if all_max and all_min:
        return "flach"
    return "trennung" if (all_max or all_min) else "keine"


@dataclass(frozen=True)
class CoxFit:
    beta: float
    se: float
    ok: bool
    status: str       # Schlüssel von STATUS_TEXT
    iterations: int
    loglik: float

    @property
    def message(self) -> str:
        return STATUS_TEXT[self.status]

    @property
    def hazard_ratio(self) -> float:
        return math.exp(self.beta) if self.ok else float("nan")


def fit_cox(time, event, x) -> CoxFit:
    """Cox-Anpassung mit einem Merkmal. Gibt nie eine stille Zahl: ohne Konvergenz sind β und Standardfehler nan, und `status` nennt den Grund."""
    t, e, _ = K.check_data(time, event)   # ungültige Eingabe (Zeit ≤ 0, nan, Länge) wirft SurvivalError, bevor eine der Prüfungen unten etwas anderes meldet
    xv = np.asarray(x, float)
    if xv.shape != t.shape or not np.all(np.isfinite(xv)):
        raise K.SurvivalError("Das Merkmal muss zu jeder Beobachtung endlich angegeben werden.")
    nan = float("nan")
    if not e.any():
        return CoxFit(nan, nan, False, "ohne_ereignis", 0, nan)
    kind = separation_kind(t, e, xv)
    if float(np.ptp(xv)) == 0.0 or kind == "flach":
        return CoxFit(nan, nan, False, "singulaer", 0, nan)
    if kind == "trennung":
        # Die Trennung wird vor dem Kern geprüft und nicht an dessen Konvergenzflag abgelesen: Bei einem einzigen Ereignis (oder wenn die
        # Log-Likelihood in Gleitkomma schon bei 0 liegt) meldete eine frühere Fassung des Kerns „konvergiert“ mit β̂ = 16 und Standardfehler 9,5·10^7;
        # die aktuelle Kopie fängt das mit einer Plausibilitätsgrenze ab (ok = False). Die Vorabprüfung gilt unabhängig davon (tests/test_oracle_cox.py).
        return CoxFit(nan, nan, False, "trennung", 0, nan)
    r = K.cox_fit(t, e, xv)
    if r["ok"]:
        return CoxFit(float(r["beta"][0]), float(r["se"][0]), True, "ok", int(r["iterations"]), float(r["loglik"]))
    return CoxFit(nan, nan, False, "singulaer" if "singulär" in r["message"] else "keine_konvergenz", int(r["iterations"]), nan)


# ------------------------------------------------------------------------------------------------ Regression auf die Beobachtungszeit


def ols_slope(y, x, mask=None) -> float:
    """Steigung der Kleinste-Quadrate-Geraden y ~ 1 + x (optional nur über die Zeilen mit mask); nan, wenn weniger als 3 Zeilen oder x konstant."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if mask is not None:
        y, x = y[mask], x[mask]
    if y.size < 3 or float(np.ptp(x)) == 0.0:
        return float("nan")
    xc = x - x.mean()
    return float(np.sum(xc * (y - y.mean())) / np.sum(xc * xc))


# ------------------------------------------------------------------------------------------------ Breslow-Grundrate und Gruppenkurven


def breslow_baseline(time, event, x, beta: float):
    """Kumulierte Grundrate nach Breslow: Ĥ0(t) = Σ_{t_k ≤ t} d_k / Σ_{j ∈ R_k} exp(β·x_j). Gibt Ereigniszeiten und Ĥ0 an diesen Zeiten zurück."""
    t, e, xv = np.asarray(time, float), np.asarray(event, bool), np.asarray(x, float)
    ts = np.unique(t[e])
    w = np.exp(beta * xv)
    inc = np.array([float((e & (t == tk)).sum()) / float(w[_risk_set(t, tk)].sum()) for tk in ts])
    return ts, np.cumsum(inc)


def group_of(x) -> np.ndarray:
    """Gruppe 0, 1, 2 nach dem Merkmal (Grenzen = Terzile der Standardnormalverteilung)."""
    return np.digitize(np.asarray(x, float), C.GROUP_CUTS)


def baseline_at(ts, h0, grid) -> np.ndarray:
    """Treppenfunktion Ĥ0 an den Zeiten `grid` (vor der ersten Ereigniszeit 0)."""
    idx = np.searchsorted(ts, np.asarray(grid, float), side="right") - 1
    return np.where(idx < 0, 0.0, h0[np.maximum(idx, 0)]) if ts.size else np.zeros(len(grid))


def predicted_group_survival(ts, h0, beta: float, x_group, grid) -> np.ndarray:
    """Vorhergesagte Überlebenskurve einer Gruppe: Mittel der Kurven exp(−Ĥ0(t)·exp(β̂·x_i)) über ihre Mitglieder (nicht die Kurve beim mittleren x)."""
    h = baseline_at(ts, h0, grid)
    return np.exp(-np.outer(h, np.exp(beta * np.asarray(x_group, float)))).mean(axis=1)


def comparison_limit(sample: Sample, groups) -> float:
    """Vergleichsbereich bis τ: die letzte Ereigniszeit, zu der in jeder Gruppe noch mindestens MIN_AT_RISK Container im Risiko sind (kleinste über die Gruppen);
    nan, wenn eine Gruppe nie so viele im Risiko hat."""
    limits = []
    for g in range(len(C.GROUP_CUTS) + 1):
        m = groups == g
        km = K.kaplan_meier(sample.time[m], sample.event[m])
        ok = km["n"] >= C.MIN_AT_RISK
        if not ok.any():
            return float("nan")
        limits.append(float(km["t"][ok].max()))
    return min(limits)


@dataclass(frozen=True)
class CurveComparison:
    tau: float                 # Vergleichsbereich 0 bis tau
    gaps: tuple[float, ...]    # größter Abstand |Cox − Kaplan–Meier| je Gruppe im Bereich, in Wahrscheinlichkeit (0 bis 1)
    truth_gaps: tuple[float, ...]   # größter Abstand |Cox − Wahrheit| je Gruppe im Bereich
    km_truth_gaps: tuple[float, ...]  # größter Abstand |Kaplan–Meier − Wahrheit| je Gruppe im Bereich


def compare_group_curves(sample: Sample, world: World, fit: CoxFit) -> CurveComparison | None:
    """Größter Abstand der Cox-Kurve (Breslow-Grundrate) zur Kaplan–Meier-Kurve der Gruppe und zur wahren Kurve, je Gruppe, im Bereich bis τ.
    None, wenn kein Vergleichsbereich existiert oder die Anpassung nicht konvergiert ist."""
    if not fit.ok:
        return None
    groups = group_of(sample.x)
    tau = comparison_limit(sample, groups)
    if not tau > 0:
        return None
    ts, h0 = breslow_baseline(sample.time, sample.event, sample.x, fit.beta)
    grid = ts[ts <= tau]
    gaps, truth_gaps, km_truth = [], [], []
    for g in range(len(C.GROUP_CUTS) + 1):
        m = groups == g
        km = K.kaplan_meier(sample.time[m], sample.event[m])
        s_km = np.array([K.survival_at(km, float(v)) for v in grid])
        s_cox = predicted_group_survival(ts, h0, fit.beta, sample.x[m], grid)
        s_true = np.mean([true_survival(grid, xi, world.beta, world.shape) for xi in sample.x[m]], axis=0)
        gaps.append(float(np.max(np.abs(s_cox - s_km))))
        truth_gaps.append(float(np.max(np.abs(s_cox - s_true))))
        km_truth.append(float(np.max(np.abs(s_km - s_true))))
    return CurveComparison(tau, tuple(gaps), tuple(truth_gaps), tuple(km_truth))


# ------------------------------------------------------------------------------------------------ Simulationsstudie (Experiment 2)


@dataclass(frozen=True)
class Study:
    """Ergebnisse über `reps` Läufe mit festen Seeds (Lauf r hat den Seed `run_seed(world.seed, r)`). nan: kein Schätzer in diesem Lauf."""

    beta: np.ndarray
    se: np.ndarray
    ols_all: np.ndarray
    ols_events: np.ndarray
    censored: np.ndarray     # Anteil Zensierte je Lauf
    statuses: tuple[str, ...]


def run_study(world: World, reps: int = C.REPS) -> Study:
    b, s, oa, oe, cz, st = [], [], [], [], [], []
    for r in range(reps):
        sm = simulate(world, r)
        fit = fit_cox(sm.time, sm.event, sm.x)
        logt = np.log(sm.time)
        b.append(fit.beta)
        s.append(fit.se)
        oa.append(ols_slope(logt, sm.x))
        oe.append(ols_slope(logt, sm.x, sm.event))
        cz.append(1.0 - float(sm.event.mean()))
        st.append(fit.status)
    return Study(np.array(b), np.array(s), np.array(oa), np.array(oe), np.array(cz), tuple(st))


@dataclass(frozen=True)
class Summary:
    reps: int
    n_ok: int                    # Läufe mit Cox-Schätzer
    n_no_fit: int                # Läufe ohne Schätzer (Trennung, ohne Ereignis, …)
    status_counts: tuple[tuple[str, int], ...]
    censored: float              # mittlerer Anteil Zensierte
    cox_mean: float
    cox_se_of_mean: float
    cox_sd: float                # Streuung der β̂ über die Läufe
    mean_se: float               # mittlerer modellbasierter Standardfehler
    coverage_hits: int           # Läufe, in denen β̂ ± 1,96·SE die Wahrheit enthält
    ols_all_mean: float
    ols_all_se_of_mean: float
    ols_events_mean: float
    ols_events_se_of_mean: float
    truth_cox: float
    truth_ols: float


def _mean_se(a: np.ndarray) -> tuple[float, float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), (float(a.std(ddof=1)) / math.sqrt(a.size) if a.size > 1 else float("nan"))


def summarize(study: Study, world: World) -> Summary:
    ok = np.isfinite(study.beta)
    cox_mean, cox_sem = _mean_se(study.beta)
    hits = int(np.sum(np.abs(study.beta[ok] - world.beta) <= C.Z95 * study.se[ok]))
    oa, oa_se = _mean_se(study.ols_all)
    oe, oe_se = _mean_se(study.ols_events)
    counts: dict[str, int] = {}
    for s in study.statuses:
        counts[s] = counts.get(s, 0) + 1
    return Summary(
        reps=len(study.statuses), n_ok=int(ok.sum()), n_no_fit=int((~ok).sum()), status_counts=tuple(sorted(counts.items())),
        censored=float(study.censored.mean()), cox_mean=cox_mean, cox_se_of_mean=cox_sem,
        cox_sd=float(study.beta[ok].std(ddof=1)) if ok.sum() > 1 else float("nan"),
        mean_se=float(study.se[ok].mean()) if ok.any() else float("nan"), coverage_hits=hits,
        ols_all_mean=oa, ols_all_se_of_mean=oa_se, ols_events_mean=oe, ols_events_se_of_mean=oe_se,
        truth_cox=world.beta, truth_ols=-world.beta / world.shape,
    )
