"""Kopie aus ueberlebenszeit-planung/kern/uz_core.py, Stand 2026-11-10 (nur die Funktionen dieser Demo; der Rest ist gelöscht).

Enthalten: SplitMix64 (Zufall), SurvivalError und check_data (Eingabeprüfung), Kaplan–Meier mit Eintritt (für die Gruppenkurven in
Experiment 3) und Cox-Partial-Likelihood nach Breslow mit Eintritt, Konvergenzflag und Meldung bei perfekter Trennung und Singularität.
Nur numpy. Konventionen: Risikomenge bei t = {i : Eintritt_i < t <= Zeit_i}; bei gleicher Zeit zählt das Ereignis vor der Zensierung;
Gleichstände bei Cox nach Breslow; Zensierung gilt als unabhängig vom Ereignis. Diese Datei wird nicht verändert (eine Textkorrektur
hier bräche den Vergleich mit der Quelle); Erweiterungen stehen in cxr_model.py.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------ Zufall

_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig.
    Alle Verteilungen sind hier selbst aus Gleichverteilten gebaut (nie numpy-Verteilungen, deren Algorithmen sich zwischen Versionen ändern)."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def normal(self, size: int) -> np.ndarray:
        """Standardnormal (Box-Muller, nur der Kosinus-Zweig); verbraucht 2·size Gleichverteilte."""
        u1 = self.uniform(size)
        u2 = self.uniform(size)
        return np.sqrt(-2.0 * np.log1p(-u1)) * np.cos(2.0 * np.pi * u2)

    def exponential(self, size: int) -> np.ndarray:
        """Exponentialverteilt mit Mittel 1 (Inversion)."""
        return -np.log1p(-self.uniform(size))

    def lognormal(self, size: int, median: float, sigma: float) -> np.ndarray:
        """Lognormal mit gegebenem Median und σ der Log-Skala."""
        return median * np.exp(sigma * self.normal(size))

    def weibull(self, size: int, shape: float, scale: float) -> np.ndarray:
        """Weibull mit Form k und Skala λ, S(t) = exp(−(t/λ)^k) (Inversion)."""
        return scale * self.exponential(size) ** (1.0 / shape)



# ------------------------------------------------------------------------------------------------ Eingabeprüfung


class SurvivalError(ValueError):
    """Ungültige Eingabe (verständliche Meldung für die App)."""


def check_data(time, event, entry=None):
    """Prüft und wandelt in float-/bool-Arrays. Zeiten > 0 und endlich, Ereignis 0/1, Eintritt >= 0 und < Zeit (für jede Zeile)."""
    t = np.asarray(time, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise SurvivalError("Es braucht mindestens eine Beobachtung (eindimensionale Liste der Zeiten).")
    e_raw = np.asarray(event)
    if e_raw.shape != t.shape:
        raise SurvivalError("Zeiten und Ereignisse müssen gleich viele Einträge haben.")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise SurvivalError("Alle Zeiten müssen endlich und größer als 0 sein.")
    if not np.all(np.isin(e_raw, (0, 1, False, True))):
        raise SurvivalError("Das Ereignis muss 0 (zensiert) oder 1 (Ereignis) sein.")
    e = e_raw.astype(bool)
    if entry is None:
        en = np.zeros_like(t)
    else:
        en = np.asarray(entry, dtype=float)
        if en.shape != t.shape or not np.all(np.isfinite(en)) or np.any(en < 0):
            raise SurvivalError("Der Eintritt muss zu jeder Beobachtung angegeben werden, endlich und nicht negativ sein.")
        bad = np.where(en >= t)[0]
        if bad.size:
            raise SurvivalError(f"Der Eintritt muss vor der Zeit liegen (Beobachtung {int(bad[0]) + 1}: Eintritt {en[bad[0]]:g} ≥ Zeit {t[bad[0]]:g}).")
    return t, e, en


# ------------------------------------------------------------------------------------------------ Kaplan–Meier


def kaplan_meier(time, event, entry=None) -> dict:
    """Kaplan–Meier-Schätzer mit verzögertem Eintritt. Gibt Arrays je Ereigniszeit zurück:
    t, n (im Risiko), d (Ereignisse), s (Überleben nach t), gw (Greenwood-Summe Σ d/(n(n−d)), Terme mit n = d fehlen: dort ist s = 0).
    """
    t, e, en = check_data(time, event, entry)
    ts = np.unique(t[e])
    ts_sorted = np.sort(t)
    en_sorted = np.sort(en)
    n = np.searchsorted(en_sorted, ts, side="left") - np.searchsorted(ts_sorted, ts, side="left")  # #(Eintritt < t) − #(Zeit < t)
    d = np.array([np.sum((t == x) & e) for x in ts], dtype=float) if ts.size else np.zeros(0)
    n = n.astype(float)
    if np.any(n < d):
        raise SurvivalError("Interner Fehler: mehr Ereignisse als Beobachtungen im Risiko.")
    s = np.cumprod(1.0 - d / n) if ts.size else np.zeros(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(n > d, d / (n * (n - d)), 0.0)
    return {"t": ts, "n": n, "d": d, "s": s, "gw": np.cumsum(terms), "n_obs": int(t.size)}


def _index_at(km: dict, x: float) -> int:
    return int(np.searchsorted(km["t"], x, side="right")) - 1


def survival_at(km: dict, x: float) -> float:
    """Ŝ(x); vor der ersten Ereigniszeit 1."""
    i = _index_at(km, x)
    return 1.0 if i < 0 else float(km["s"][i])


# ------------------------------------------------------------------------------------------------ Cox


def _cox_sums(start, stop, x, beta, ts):
    """Gewichtete Summen s0, s1, s2 über die Risikomengen (start < t <= stop) an den Zeiten ts. Gewichte werden vorher um das Maximum verschoben."""
    eta = x @ beta
    shift = float(eta.max())
    w = np.exp(eta - shift)
    wx = w[:, None] * x
    wxx = w[:, None, None] * x[:, :, None] * x[:, None, :]
    os_, ob = np.argsort(stop), np.argsort(start)
    i_stop = np.searchsorted(stop[os_], ts, side="left")
    i_start = np.searchsorted(start[ob], ts, side="left")

    def suffix(a, order):
        c = np.cumsum(a[order][::-1], axis=0)[::-1]
        return np.concatenate([c, np.zeros((1,) + a.shape[1:])], axis=0)

    s0 = suffix(w, os_)[i_stop] - suffix(w, ob)[i_start]
    s1 = suffix(wx, os_)[i_stop] - suffix(wx, ob)[i_start]
    s2 = suffix(wxx, os_)[i_stop] - suffix(wxx, ob)[i_start]
    return s0, s1, s2, shift


def cox_fit(time, event, x, entry=None, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Cox-Partial-Likelihood (Breslow), Newton mit Schrittweitenhalbierung. x: (n, p) oder (n,).
    Gibt beta, se (aus der beobachteten Information), loglik, iterations, ok, message. ok = False bei Nichtkonvergenz (häufig perfekte Trennung:
    es gibt dann keinen endlichen Schätzer) oder singulärer Information (konstantes oder kollineares Merkmal, alle zensiert).
    Merkmale vorher zentrieren und skalieren, wenn sie sehr große Werte haben (Genauigkeit der Risikomengen-Summen)."""
    t, e, en = check_data(time, event, entry)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != t.size or not np.all(np.isfinite(x)):
        raise SurvivalError("Die Merkmale müssen zu jeder Beobachtung endlich angegeben werden.")
    p = x.shape[1]
    ts, inv = np.unique(t[e], return_inverse=True)
    if ts.size == 0:
        return {"beta": np.full(p, np.nan), "se": np.full(p, np.nan), "loglik": float("nan"), "iterations": 0, "ok": False, "message": "Ohne Ereignis gibt es keinen Schätzer."}
    dcount = np.bincount(inv, minlength=ts.size).astype(float)
    sumx = np.stack([np.bincount(inv, weights=x[e, j], minlength=ts.size) for j in range(p)], axis=1)

    def evaluate(beta):
        s0, s1, s2, shift = _cox_sums(en, t, x, beta, ts)
        if np.any(s0 <= 0):
            return None
        m = s1 / s0[:, None]
        ll = float(np.sum(sumx @ beta) - np.sum(dcount * (np.log(s0) + shift)))
        grad = (sumx - dcount[:, None] * m).sum(axis=0)
        hess = np.zeros((p, p))
        for k in range(ts.size):
            hess += dcount[k] * (s2[k] / s0[k] - np.outer(m[k], m[k]))
        return ll, grad, hess

    beta = np.zeros(p)
    cur = evaluate(beta)
    if cur is None:
        return {"beta": np.full(p, np.nan), "se": np.full(p, np.nan), "loglik": float("nan"), "iterations": 0, "ok": False, "message": "Die Risikomengen sind leer."}
    ll, grad, hess = cur
    for it in range(1, max_iter + 1):
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return {"beta": np.full(p, np.nan), "se": np.full(p, np.nan), "loglik": ll, "iterations": it, "ok": False,
                    "message": "Die Information ist singulär (konstantes oder kollineares Merkmal, oder zu wenige Ereignisse)."}
        scale = 1.0
        for _ in range(40):
            new = evaluate(beta + scale * step)
            if new is not None and new[0] >= ll - 1e-12:
                break
            scale *= 0.5
        else:
            break
        beta = beta + scale * step
        converged = abs(new[0] - ll) < tol * (1.0 + abs(ll)) and np.max(np.abs(scale * step)) < 1e-7
        ll, grad, hess = new
        if converged:
            try:
                cov = np.linalg.inv(hess)
            except np.linalg.LinAlgError:
                break
            se = np.sqrt(np.abs(np.diag(cov)))
            # „Konvergenz“ bei riesigem Standardfehler oder riesigem β̂ ist eine verkappte perfekte Trennung (z. B. ein einziges Ereignis):
            # der Schätzer bleibt nur wegen der Abbruchtoleranz stehen. Dann kein Ergebnis melden.
            if not np.all(np.isfinite(se)) or np.any(se > 50.0) or np.max(np.abs(beta)) > 20.0:
                break
            return {"beta": beta, "se": se, "loglik": ll, "iterations": it, "ok": True, "message": ""}
        if np.max(np.abs(beta)) > 25:
            break
    return {"beta": beta, "se": np.full(p, np.nan), "loglik": ll, "iterations": it, "ok": False,
            "message": "Keine Konvergenz; bei perfekter Trennung der Gruppen (alle Ereignisse in einer Gruppe zuerst) gibt es keinen endlichen Schätzer."}
