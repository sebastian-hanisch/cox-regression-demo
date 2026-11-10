"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cxr_constants as C
import cxr_core as K
import cxr_model as M


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# ------------------------------------------------------------------------------------------------ Experiment 1


def build_loglik(time, event, x, path, probe: float, fit: M.CoxFit):
    """Partielle Log-Likelihood über β für die vier Container; Newton-Schritte als Punkte, β zum Ausprobieren als Raute, β̂ als senkrechte Linie (nur mit Schätzer)."""
    betas = list(C.BETA_GRID)
    ll = [M.partial_loglik(b, time, event, x) for b in betas]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=betas, y=ll, mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="log L(β)", hovertemplate="β = %{x:.2f}: %{y:.4f}<extra></extra>"))
    inside = [p for p in path if C.BETA_GRID[0] <= p["beta"] <= C.BETA_GRID[-1]]
    fig.add_trace(go.Scatter(x=[p["beta"] for p in inside], y=[p["loglik"] for p in inside], mode="markers+text", text=[str(i) for i in range(len(inside))],
                             textposition="top center", marker=dict(size=9, color=C.COLOR_COX), name="Newton-Schritte (0 = Start bei β = 0)",
                             hovertemplate="β = %{x:.4f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[probe], y=[M.partial_loglik(probe, time, event, x)], mode="markers", marker=dict(size=13, symbol="diamond", color=C.COLOR_OLS),
                             name="β zum Ausprobieren", hovertemplate="β = %{x:.2f}: %{y:.4f}<extra></extra>"))
    if fit.ok:
        fig.add_vline(x=fit.beta, line=dict(color=C.COLOR_COX, dash="dash"), annotation_text=f"β̂ = {fit.beta:.4f}", annotation_position="bottom right")
    fig.update_xaxes(title_text="β")
    fig.update_yaxes(title_text="partielle Log-Likelihood")
    return _base(fig)


# ------------------------------------------------------------------------------------------------ Experiment 2


def build_estimators(s: M.Summary):
    """Links: Cox β̂ gegen die Wahrheit β. Rechts: Steigung der Regression von log(beobachtete Zeit) auf das Merkmal gegen die Wahrheit −β/k.
    Fehlerbalken: ± 1 Standardfehler des Mittels über die Läufe."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Cox: β̂ gegen β", "Regression auf die Zeit: Steigung gegen −β/k"), horizontal_spacing=0.12)
    fig.add_trace(go.Bar(x=["Cox β̂", "Wahrheit β"], y=[s.cox_mean, s.truth_cox], marker_color=[C.COLOR_COX, C.COLOR_TRUTH],
                         error_y=dict(type="data", array=[s.cox_se_of_mean, 0], visible=True), showlegend=False,
                         hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=["Regression, alle", "Regression, nur Ereignisse", "Wahrheit −β/k"], y=[s.ols_all_mean, s.ols_events_mean, s.truth_ols],
                         marker_color=[C.COLOR_OLS, C.COLOR_OLS, C.COLOR_TRUTH], error_y=dict(type="data", array=[s.ols_all_se_of_mean, s.ols_events_se_of_mean, 0], visible=True),
                         showlegend=False, hovertemplate="%{x}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(title_text="Schätzwert", row=1, col=1)
    return _base(fig, height=380)


def build_intervals(study: M.Study, world, shown: int = 40):
    """Die ersten Läufe: β̂ mit Intervall β̂ ± 1,96·SE; rot, wenn das Intervall die Wahrheit β nicht enthält; Läufe ohne Schätzer fehlen."""
    idx = [i for i in range(min(shown, len(study.beta))) if np.isfinite(study.beta[i])]
    hit = [abs(study.beta[i] - world.beta) <= C.Z95 * study.se[i] for i in idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i + 1 for i in idx], y=[study.beta[i] for i in idx], mode="markers",
                             marker=dict(size=6, color=[C.COLOR_COX if h else C.COLOR_OLS for h in hit]),
                             error_y=dict(type="data", array=[C.Z95 * study.se[i] for i in idx], visible=True, thickness=1.2),
                             showlegend=False, hovertemplate="Lauf %{x}: β̂ = %{y:.3f}<extra></extra>"))
    fig.add_hline(y=world.beta, line=dict(color=C.COLOR_TRUTH, dash="dash"), annotation_text=f"Wahrheit β = {world.beta:g}", annotation_position="top left")
    fig.update_xaxes(title_text=f"Lauf (die ersten {shown})")
    fig.update_yaxes(title_text="β̂ mit 95-%-Intervall")
    return _base(fig, height=320)


# ------------------------------------------------------------------------------------------------ Experiment 3


def build_group_curves(sample, world, fit: M.CoxFit, tau: float):
    """Je Gruppe nach dem Merkmal: Kaplan–Meier-Kurve (Treppe) und Cox-Kurve mit Breslow-Grundrate (gestrichelt), beide bis zur größten Beobachtungszeit; der Vergleichsbereich bis τ ist markiert."""
    groups = M.group_of(sample.x)
    ts, h0 = M.breslow_baseline(sample.time, sample.event, sample.x, fit.beta)
    grid = np.concatenate([[0.0], ts])
    labels = ("niedriges Merkmal", "mittleres Merkmal", "hohes Merkmal")
    fig = go.Figure()
    for g in range(3):
        m = groups == g
        km = K.kaplan_meier(sample.time[m], sample.event[m])
        t_km = np.concatenate([[0.0], km["t"]])
        s_km = np.concatenate([[1.0], km["s"]])
        fig.add_trace(go.Scatter(x=t_km, y=s_km, mode="lines", line=dict(color=C.GROUP_COLORS[g], width=2, shape="hv"), name=f"{labels[g]}: Kaplan–Meier",
                                 hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"))
        s_cox = M.predicted_group_survival(ts, h0, fit.beta, sample.x[m], grid)
        fig.add_trace(go.Scatter(x=grid, y=s_cox, mode="lines", line=dict(color=C.GROUP_COLORS[g], width=2, dash="dash", shape="hv"), name=f"{labels[g]}: Cox",
                                 hovertemplate="t = %{x:.2f}: %{y:.3f}<extra></extra>"))
    if tau > 0:
        fig.add_vline(x=tau, line=dict(color=C.COLOR_TRUTH, dash="dot"), annotation_text="Vergleichsbereich endet", annotation_position="top left")
    fig.update_xaxes(title_text="Zeit (Tage)")
    fig.update_yaxes(title_text="Anteil noch nicht abgeholt", range=[0, 1.02])
    return _base(fig, height=420)
