"""Messskript der Demo: jede Zahl der README steht in `tools/measure_cox_result.json` (eingefroren, feste Seeds).

Aufruf (Repo-Wurzel):  _venvs/test/Scripts/python.exe -X utf8 tools/measure_cox.py            schreibt tools/measure_cox_result.json
                       _venvs/test/Scripts/python.exe -X utf8 tools/measure_cox.py --print    zeigt die Zahlen, schreibt nichts

Alle Messungen laufen mit dem Code der Demo (SplitMix64, cxr_model); Läufe je Zelle: C.REPS = 200, Lauf r hat den Seed `run_seed(Zufalls-Seed, r)`.
Die Funktionen sind rein, die Tests (tests/test_claims.py) rechnen Teile davon neu.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cxr_constants as C  # noqa: E402
import cxr_model as M  # noqa: E402
import cxr_scenario as S  # noqa: E402

RESULT = Path(__file__).with_name("measure_cox_result.json")
STANDARD = S.World()            # n = 300, Fenster 12 Tage, β = 0,7, Form 1,5, Zufalls-Seed 35
SEEDS_STABILITY = (1, 2, 3, 4, 5)


def cell(world: S.World, reps: int = C.REPS) -> dict:
    """Eine Zelle: alle Kennzahlen der Läufe als Zahlen (JSON-fähig)."""
    study = M.run_study(world, reps)
    s = M.summarize(study, world)
    return {
        "n": world.n, "window": world.window, "beta": world.beta, "shape": world.shape, "seed": world.seed, "reps": s.reps,
        "censored": s.censored, "n_ok": s.n_ok, "n_no_fit": s.n_no_fit, "status_counts": dict(s.status_counts),
        "cox_mean": s.cox_mean, "cox_se_of_mean": s.cox_se_of_mean, "cox_sd": s.cox_sd, "mean_se": s.mean_se, "coverage_hits": s.coverage_hits,
        "ols_all_mean": s.ols_all_mean, "ols_all_se_of_mean": s.ols_all_se_of_mean,
        "ols_events_mean": s.ols_events_mean, "ols_events_se_of_mean": s.ols_events_se_of_mean,
        "truth_cox": s.truth_cox, "truth_ols": s.truth_ols,
    }


def curve_cell(world: S.World, reps: int = C.REPS) -> dict:
    """Experiment 3 über `reps` Datensätze: größter Abstand Cox-Kurve gegen Kaplan–Meier je Datensatz (Maximum über die drei Gruppen), dazu die Abstände gegen die Wahrheit."""
    gap, truth_gap, km_truth, taus, skipped = [], [], [], [], 0
    for r in range(reps):
        sm = S.simulate(world, r)
        fit = M.fit_cox(sm.time, sm.event, sm.x)
        cmp_ = M.compare_group_curves(sm, world, fit)
        if cmp_ is None:
            skipped += 1
            continue
        gap.append(max(cmp_.gaps))
        truth_gap.append(max(cmp_.truth_gaps))
        km_truth.append(max(cmp_.km_truth_gaps))
        taus.append(cmp_.tau)
    q = lambda a, p: float(np.quantile(np.array(a), p))  # noqa: E731
    return {
        "n": world.n, "window": world.window, "reps": reps, "skipped": skipped,
        "gap_median": q(gap, 0.5), "gap_q90": q(gap, 0.9), "gap_max": float(max(gap)),
        "truth_gap_median": q(truth_gap, 0.5), "km_truth_gap_median": q(km_truth, 0.5), "tau_median": q(taus, 0.5),
    }


def hand_cell() -> dict:
    """Das Vier-Container-Handbeispiel (x = 1, 0, 1, 0, Zeiten 1 bis 4)."""
    t, e, x = np.array(S.HAND_TIMES), np.ones(4, bool), np.array(S.PATTERNS[S.DEFAULT_PATTERN])
    fit = M.fit_cox(t, e, x)
    path = M.newton_path(t, e, x)
    u, i = M.score_info(0.0, t, e, x)
    return {"score0": u, "info0": i, "beta_hat": fit.beta, "se": fit.se, "hazard_ratio": fit.hazard_ratio, "newton1": path[1]["beta"], "loglik0": path[0]["loglik"]}


def measure() -> dict:
    out: dict = {"reps": C.REPS}
    out["standard"] = cell(STANDARD)                                                   # Messung V6: n = 300, 55 % zensiert
    out["stability"] = [cell(S.World(seed=s)) for s in SEEDS_STABILITY]                # derselbe Aufbau, andere Zufalls-Seeds
    out["by_n"] = [cell(S.World(n=n)) for n in (10, 20, 50, 100, 300, 600)]            # Überdeckung und Läufe ohne Schätzer (V20b)
    out["by_window"] = [cell(S.World(window=w)) for w in C.WINDOW_OPTIONS]             # Zensierungsgrad
    out["by_shape"] = [cell(S.World(shape=k)) for k in (1.0, 1.5, 2.5)]                # andere Grundformen
    out["no_effect"] = cell(S.World(beta=0.0))                                         # ohne Wirkung
    out["curves"] = curve_cell(STANDARD)                                               # Experiment 3
    out["curves_small"] = curve_cell(S.World(n=100))                                   # Experiment 3 bei n = 100
    out["hand"] = hand_cell()
    return out


def tables(d: dict) -> dict[str, str]:
    """Die Markdown-Tabellen der README aus den eingefrorenen Zahlen (Dezimalkomma, echtes Minuszeichen). tests/test_claims.py prüft, dass jede Tabelle wörtlich in der README steht."""
    from cxr_evaluation import num, pct

    def ratio(a, b):
        return pct(a / b)

    s = d["standard"]
    t_standard = "\n".join([
        "| Schätzer | Mittel über 200 Läufe (± Standardfehler des Mittels) | Wahrheit | Anteil der Wahrheit |",
        "|---|---|---|---|",
        f"| Cox β̂ | {num(s['cox_mean'], 3)} ± {num(s['cox_se_of_mean'], 3)} | {num(s['truth_cox'])} | {ratio(s['cox_mean'], s['truth_cox'])} |",
        f"| Regression von log(Zeit) auf x, alle Container | {num(s['ols_all_mean'], 3)} ± {num(s['ols_all_se_of_mean'], 3)} | {num(s['truth_ols'])} | {ratio(s['ols_all_mean'], s['truth_ols'])} |",
        f"| Regression, nur Container mit Ereignis | {num(s['ols_events_mean'], 3)} ± {num(s['ols_events_se_of_mean'], 3)} | {num(s['truth_ols'])} | {ratio(s['ols_events_mean'], s['truth_ols'])} |",
    ])
    rows = ["| Zufalls-Seed | Cox β̂ | Regression, alle | Intervall enthält β |", "|---|---|---|---|"]
    for c in [s] + d["stability"]:
        rows.append(f"| {c['seed']} | {num(c['cox_mean'])} | {num(c['ols_all_mean'])} | {c['coverage_hits']} von {c['n_ok']} |")
    t_stability = "\n".join(rows)
    rows = ["| Fenster (Tage) | zensiert | Cox β̂ | Regression, alle | Regression, nur Ereignisse | Regression (alle) in % der Wahrheit |", "|---|---|---|---|---|---|"]
    for c in d["by_window"]:
        rows.append(f"| {c['window']} | {pct(c['censored'], 1)} | {num(c['cox_mean'])} | {num(c['ols_all_mean'])} | {num(c['ols_events_mean'])} | {ratio(c['ols_all_mean'], c['truth_ols'])} |")
    t_window = "\n".join(rows)
    rows = ["| n | Läufe ohne Schätzer | Cox β̂ (Läufe mit Schätzer) | mittlerer Standardfehler | Streuung der β̂ | Intervall enthält β |", "|---|---|---|---|---|---|"]
    for c in d["by_n"]:
        rows.append(f"| {c['n']} | {c['n_no_fit']} von {c['reps']} | {num(c['cox_mean'])} | {num(c['mean_se'], 3)} | {num(c['cox_sd'], 3)} | {c['coverage_hits']} von {c['n_ok']} |")
    t_n = "\n".join(rows)
    rows = ["| Form k | zensiert | Wahrheit −β/k | Cox β̂ | Regression, alle | Regression (alle) in % der Wahrheit |", "|---|---|---|---|---|---|"]
    for c in d["by_shape"]:
        rows.append(f"| {num(c['shape'], 1)} | {pct(c['censored'], 1)} | {num(c['truth_ols'])} | {num(c['cox_mean'])} | {num(c['ols_all_mean'])} | {ratio(c['ols_all_mean'], c['truth_ols'])} |")
    t_shape = "\n".join(rows)
    rows = ["| n | Datensätze (ohne Vergleichsbereich) | Median: größter Abstand Cox–Kaplan–Meier | 90-%-Quantil | Median: Kaplan–Meier gegen Wahrheit | Median: Cox gegen Wahrheit | Median Vergleichsbereich τ |", "|---|---|---|---|---|---|---|"]
    for c in (d["curves"], d["curves_small"]):
        rows.append(f"| {c['n']} | {c['reps']} ({c['skipped']}) | {num(100 * c['gap_median'], 1)} | {num(100 * c['gap_q90'], 1)} | {num(100 * c['km_truth_gap_median'], 1)} | {num(100 * c['truth_gap_median'], 1)} | {num(c['tau_median'], 1)} Tage |")
    t_curves = "\n".join(rows)
    return {"standard": t_standard, "stability": t_stability, "window": t_window, "n": t_n, "shape": t_shape, "curves": t_curves}


if __name__ == "__main__":
    t0 = time.time()
    result = measure()
    text = json.dumps(result, indent=1, ensure_ascii=False, sort_keys=True)
    if "--print" in sys.argv:
        print(text)
    else:
        RESULT.write_text(text + "\n", encoding="utf-8")
        print(f"geschrieben: {RESULT} ({time.time() - t0:.0f} s)")
