"""Kennzahlen, Texte und Verdikte (reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl im Text an handgesetzten Werten)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import cxr_constants as C
import cxr_model as M
from cxr_scenario import HAND_TIMES, PATTERNS, Sample, World, simulate


def num(x: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma und echtem Minuszeichen (für Texte)."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    """Anteil als Prozentzahl, zum Beispiel 0,456 → „46 %“."""
    return f"{num(x * 100, digits)} %"


def frac(value) -> str:
    """Bruch als Text: 1/2, −1/3, 0 (ganze Zahlen ohne Nenner)."""
    f = Fraction(value)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}".replace("-", "−")


# ------------------------------------------------------------------------------------------------ Experiment 1


@dataclass(frozen=True)
class HandAnalysis:
    pattern: str
    time: np.ndarray
    event: np.ndarray
    x: np.ndarray
    fit: M.CoxFit
    path: list
    exact_rows: list    # Risikomengen bei β = 0 in Brüchen
    probe_rows: list    # Risikomengen bei dem gewählten β
    probe: float
    probe_score: float
    probe_info: float


def analyse_hand(pattern: str, probe: float) -> HandAnalysis:
    x = np.array(PATTERNS[pattern], dtype=float)
    t = np.array(HAND_TIMES, dtype=float)
    e = np.ones(4, dtype=bool)
    u, i = M.score_info(probe, t, e, x)
    return HandAnalysis(pattern, t, e, x, M.fit_cox(t, e, x), M.newton_path(t, e, x), M.risk_rows_exact(t, e, x), M.risk_rows(probe, t, e, x), probe, u, i)


def verdict_hand(a: HandAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 1: Maximum der partiellen Likelihood oder die Meldung, warum es keines gibt."""
    if a.fit.ok:
        return "success", (f"Die partielle Log-Likelihood hat ihr Maximum bei **β̂ = {num(a.fit.beta, 4)}**; das Hazard-Verhältnis exp(β̂) ist {num(a.fit.hazard_ratio, 3)}. "
                           f"Der Standardfehler ist {num(a.fit.se, 2)}: Mit vier Containern sagt das nichts über die Welt, es dient nur zum Nachrechnen.")
    return "warning", f"**Kein Schätzer.** {a.fit.message}"


# ------------------------------------------------------------------------------------------------ Experiment 2


def verdict_study(s: M.Summary, world: World) -> tuple[str, str]:
    """Verdikt zu Experiment 2: Cox gegen Regression auf die beobachtete Zeit, jeweils mit Wahrheit, Läufen und Zensierungsgrad."""
    if s.n_ok == 0:
        return "warning", f"In keinem der {s.reps} Läufe gibt es einen Cox-Schätzer (siehe Hinweis darunter); ohne Schätzer gibt es nichts zu vergleichen."
    head = f"Über {s.reps} Läufe mit n = {world.n} Containern und Beobachtungsfenster {world.window} Tage ({pct(s.censored)} zensiert): "
    cox = f"Cox β̂ im Mittel {num(s.cox_mean)} (Wahrheit {num(s.truth_cox)}). "
    ols = (f"Die Regression von log(beobachtete Zeit) auf das Merkmal hat die Steigung {num(s.ols_all_mean)} über alle Container und {num(s.ols_events_mean)} nur über die Ereignisse "
           f"(Wahrheit −β/k = {num(s.truth_ols)}). ")
    if s.truth_ols == 0.0:
        return "info", head + cox + ols + "Ohne Wirkung ist die Wahrheit für beide 0; verglichen wird die Streuung um 0."
    ratio = s.ols_all_mean / s.truth_ols
    if ratio < 0.9:
        return "success", head + cox + ols + f"Die Regression erreicht damit **{pct(ratio)}** der Wahrheit, Cox **{pct(s.cox_mean / s.truth_cox)}**."
    return "info", head + cox + ols + f"Die Regression liegt hier bei {pct(ratio)} der Wahrheit: bei dieser Zensierung ist sie nicht deutlich zur Null verzerrt."


def coverage_text(s: M.Summary) -> str:
    """Überdeckung von β̂ ± 1,96·SE: Zählung k von m Läufen mit dem Binomial-Standardfehler dieses Anteils."""
    if s.n_ok == 0:
        return "Ohne Schätzer gibt es kein Intervall."
    p = s.coverage_hits / s.n_ok
    se = math.sqrt(p * (1 - p) / s.n_ok)
    return (f"Das Intervall β̂ ± 1,96·SE enthält die Wahrheit in **{s.coverage_hits} von {s.n_ok} Läufen** ({pct(p, 1)}; Binomial-Standardfehler dieses Anteils "
            f"{num(100 * se, 1)} Prozentpunkte). Mittlerer Standardfehler {num(s.mean_se, 3)}, tatsächliche Streuung der β̂ über die Läufe {num(s.cox_sd, 3)}.")


def failure_text(s: M.Summary) -> str:
    """Hinweis auf Läufe ohne Cox-Schätzer (leer, wenn es keine gibt): Zahl und Gründe."""
    if s.n_no_fit == 0:
        return ""
    names = {"ohne_ereignis": "ohne Ereignis", "singulaer": "Information 0", "trennung": "perfekte Trennung", "keine_konvergenz": "keine Konvergenz"}
    parts = ", ".join(f"{names.get(k, k)}: {v}" for k, v in s.status_counts if k != "ok")
    return (f"In **{s.n_no_fit} von {s.reps} Läufen** gibt es keinen endlichen Cox-Schätzer ({parts}). Diese Läufe stehen weder im Mittel noch im Diagramm; "
            "das Verfahren meldet den Grund, statt eine Zahl zu liefern.")


# ------------------------------------------------------------------------------------------------ Experiment 3


@dataclass(frozen=True)
class CurveAnalysis:
    sample: Sample
    fit: M.CoxFit
    comparison: M.CurveComparison | None


def analyse_curves(world: World) -> CurveAnalysis:
    sm = simulate(world, 0)
    fit = M.fit_cox(sm.time, sm.event, sm.x)
    return CurveAnalysis(sm, fit, M.compare_group_curves(sm, world, fit))


def verdict_curves(a: CurveAnalysis) -> tuple[str, str]:
    """Verdikt zu Experiment 3: größter Abstand zwischen Cox-Kurve (Breslow-Grundrate) und Kaplan–Meier-Kurve je Gruppe im Vergleichsbereich."""
    if not a.fit.ok:
        return "warning", f"**Kein Schätzer, keine Cox-Kurven.** {a.fit.message}"
    c = a.comparison
    if c is None:
        return "info", (f"Kein Vergleichsbereich: In mindestens einer Gruppe sind nie {C.MIN_AT_RISK} Container zugleich im Risiko. Mit mehr Containern wird der Vergleich möglich.")
    gaps = ", ".join(num(100 * g, 1) for g in c.gaps)
    return "info", (f"Bis Tag {num(c.tau, 1)} (in jeder Gruppe noch mindestens {C.MIN_AT_RISK} Container im Risiko) liegt die Cox-Kurve höchstens "
                    f"**{num(100 * max(c.gaps), 1)} Prozentpunkte** neben der Kaplan–Meier-Kurve ihrer Gruppe (je Gruppe {gaps}). Zum Vergleich weicht die Kaplan–Meier-Kurve "
                    f"selbst um höchstens {num(100 * max(c.km_truth_gaps), 1)} Prozentpunkte von der wahren Kurve ab, die Cox-Kurve um höchstens {num(100 * max(c.truth_gaps), 1)}.")
