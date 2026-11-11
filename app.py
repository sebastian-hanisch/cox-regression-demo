"""Cox-Regression: Merkmale ohne Form - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Merkmal vervielfacht die Abholrate eines Containers; die Cox-Regression schätzt den Faktor, ohne die Grundform der Rate zu kennen.
Drei Experimente: die partielle Likelihood von Hand (vier Container), Cox gegen die Regression auf die beobachtete Zeit (Simulation mit
Zensierung), die Grundrate nachträglich (Breslow) gegen die Kaplan-Meier-Kurven der Gruppen.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import cxr_constants as C
import cxr_evaluation as ev
import cxr_model as M
import cxr_visualization as V
from cxr_layout import dunkles_design, kopf_css
from cxr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from cxr_scenario import PATTERNS, World

st.set_page_config(page_title="Cox-Regression – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _study(world: World):
    study = M.run_study(world, C.REPS)
    return study, M.summarize(study, world)


@st.cache_data(show_spinner=False)
def _curves(world: World):
    return ev.analyse_curves(world)


def md_table(headers, rows) -> str:
    """Markdown-Tabelle aus Texten (kein | in den Zellen)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in cxr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎚️ Cox-Regression: Merkmale ohne Form")
st.markdown(
    """
Ein **Merkmal** eines Containers verändert, wie schnell er abgeholt wird. Die **Cox-Regression** schätzt das **Verhältnis**, mit dem das Merkmal die Abholrate vervielfacht,
**ohne die Grundform der Rate zu kennen**; Container, die bei Fensterende noch nicht abgeholt sind (**zensiert**), gehen über die **Risikomengen** ein. Eine einfache Regression
auf die beobachtete Zeit behandelt dagegen jede Zensierzeit wie eine Abholzeit.

**Annahme in dieser Demo: Die Zensierung ist unabhängig vom Ereignis** (ein zensierter Container hätte dieselbe Abholrate wie ein gleicher Container, der noch im Risiko bleibt);
die Welt unten erfüllt sie, geprüft wird sie hier nicht.

Weiter unten: **🔬 Experiment 1** (Handrechnung mit vier Containern), **❓ Kernfrage mit Experiment 2** (Cox gegen Regression auf die Zeit), **🔬 Experiment 3** (die Grundrate
nachträglich), **🚧 Wo die Annahmen enden** sowie die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Cox-Regression", expanded=True):
    st.markdown(
        """
1. **Rate statt Dauer.** Für einen Container mit Merkmal $x$ gilt $h(t \\mid x) = h_0(t)\\,e^{\\beta x}$: Die Abholrate ist eine unbekannte Grundrate $h_0(t)$, mal ein Faktor $e^{\\beta x}$.
   Gesucht ist $\\beta$ (genauer das Verhältnis $e^{\\beta}$ je Einheit des Merkmals), nicht die Form von $h_0$.
2. **Risikomenge.** Wird zur Zeit $t$ ein Container abgeholt, steht sein Merkmal gegen die Merkmale aller Container, die zu $t$ noch da sind. Die Grundrate $h_0(t)$ kürzt sich
   in diesem Vergleich heraus; übrig bleibt eine Wahrscheinlichkeit, die nur von $\\beta$ abhängt.
3. **Partielle Likelihood.** Das Produkt dieser Wahrscheinlichkeiten über alle Abholungen wird in $\\beta$ maximiert (Newton-Verfahren: Score geteilt durch Information).
4. **Zensierte Container** zählen nie als Abholung, bleiben aber in den Risikomengen, solange sie beobachtet sind. Eine Regression auf die beobachtete Zeit kennt diesen Unterschied nicht.
5. **Grundrate nachträglich.** Aus $\\hat\\beta$ lässt sich die kumulierte Grundrate schätzen (Breslow); erst dann gibt es Überlebenskurven.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt der Daten** (Experimente 2 und 3)")
    beta = st.slider("Wahres β (Wirkung des Merkmals)", *bounds("beta_slider"), step=C.BETA_STEP, key="beta_slider",
                     help="Das Merkmal vervielfacht die Abholrate mit exp(β) je Einheit. 0 heißt: keine Wirkung.")
    shape = st.slider("Form k der Weibull-Grundrate", *bounds("shape_slider"), step=C.SHAPE_STEP, key="shape_slider",
                      help="k = 1: die Grundrate bleibt gleich; k > 1: sie steigt mit der Zeit; k < 1: sie fällt. Das Cox-Verfahren schätzt die Form nicht.")
    seed = st.number_input("Zufalls-Seed der Daten", *bounds("seed_input"), key="seed_input", step=1,
                           help="Jeder der Läufe in Experiment 2 hat einen eigenen, festen Zufallsstrom, der aus diesem Zufalls-Seed abgeleitet ist.")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

# --- Experiment 1: partielle Likelihood von Hand --------------------------------------------------------------------------------------------------

st.markdown("## 🧮 Das Verfahren von Hand: vier Container")
st.subheader("🔬 Experiment 1: Die partielle Likelihood")
st.markdown(
    """
Vier Container werden nacheinander zu den Zeiten 1, 2, 3 und 4 abgeholt (alle vier sind beobachtet). Jeder hat ein Merkmal $x$ (0 oder 1). Bei jeder Abholung steht das Merkmal
des abgeholten Containers gegen das **gewichtete Mittel $m$** der Merkmale aller, die dann noch da sind (die Risikomenge); bei $\\beta = 0$ ist das das einfache Mittel.
Der **Beitrag zum Score** ist $x - m$, der **Beitrag zur Information** die Varianz der Merkmale in der Risikomenge. Wähle ein Muster und ein $\\beta$ zum Ausprobieren.
"""
)
c1, c2 = st.columns(2)
pattern = c1.selectbox("Muster des Merkmals der vier Container", list(PATTERNS), key="pattern_select",
                       help="Die vier Werte gehören zu den Containern, die zu den Zeiten 1, 2, 3, 4 abgeholt werden.")
probe = c2.slider("β zum Ausprobieren", *bounds("probe_slider"), step=C.PROBE_STEP, key="probe_slider",
                  help="Die Raute im Diagramm zeigt die partielle Log-Likelihood an diesem β; die zweite Tabelle rechnet Mittel, Score und Information dort.")
hand = ev.analyse_hand(pattern, float(probe))

ex_rows = [
    [f"{r['t']:g}", ", ".join(str(v) for v in r["risk_x"]), ev.frac(r["m"]), ev.frac(r["score"]), ev.frac(r["info"])]
    for r in hand.exact_rows
]
score_sum = sum(r["score"] for r in hand.exact_rows)
info_sum = sum(r["info"] for r in hand.exact_rows)
ex_rows.append(["**Summe**", "", "", f"**{ev.frac(score_sum)}**", f"**{ev.frac(info_sum)}**"])
st.markdown("**Bei β = 0, in Brüchen**")
st.markdown(md_table(["Zeit t", "Merkmale der Risikomenge", "Mittel m", "Beitrag zum Score", "Beitrag zur Information"], ex_rows))
if info_sum > 0:
    st.caption(f"Erster Newton-Schritt von β = 0 aus: Score geteilt durch Information = {ev.frac(score_sum / info_sum)} ≈ {ev.num(float(score_sum / info_sum), 3)}.")
else:
    st.caption("Die Information ist 0: ein Newton-Schritt ist nicht möglich.")

m1, m2, m3 = st.columns(3)
m1.metric(f"Log-Likelihood bei β = {ev.num(hand.probe, 2)}", ev.num(M.partial_loglik(hand.probe, hand.time, hand.event, hand.x), 4))
m2.metric("Score U(β)", ev.num(hand.probe_score, 4))
m3.metric("Information I(β)", ev.num(hand.probe_info, 4))
with st.expander("Tabelle der Risikomengen bei dem gewählten β"):
    pr_rows = [[f"{r['t']:g}", ", ".join(f"{v:g}" for v in r["risk_x"]), ev.num(r["m"], 4), ev.num(r["score"], 4), ev.num(r["info"], 4)] for r in hand.probe_rows]
    st.markdown(md_table(["Zeit t", "Merkmale der Risikomenge", "gewichtetes Mittel m(β)", "Beitrag zum Score", "Beitrag zur Information"], pr_rows))

st.plotly_chart(V.build_loglik(hand.time, hand.event, hand.x, hand.path, hand.probe, hand.fit), width="stretch", key="loglik_chart")
st.caption("Die Newton-Schritte (Punkte, 0 = Start bei β = 0) laufen ohne Schrittweitenhalbierung, so wie man sie von Hand rechnet; die Anpassung der Demo (cox_fit) halbiert Schritte bei Bedarf.")
kind, text = ev.verdict_hand(hand)
getattr(st, kind)(text)
if not hand.fit.ok:
    st.caption("Der Kern meldet das über sein Konvergenzflag: `ok = False` mit Grund; die App zeigt dann nie eine Zahl für β̂.")

st.markdown("---")

# --- Kernfrage und Experiment 2: Cox gegen Regression auf die Zeit ---------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie stark verändert ein Merkmal die Rate, und wie weit führt die einfache Regression in die Irre?")
st.subheader("🔬 Experiment 2: Cox gegen Regression auf die beobachtete Zeit")
st.markdown(
    f"""
Welt: Weibull-Grundrate mit Skala {C.WEIBULL_SCALE:g} Tagen, ein standardnormalverteiltes Merkmal $x$, Rate mal $e^{{\\beta x}}$ (β und Form $k$ in der Seitenleiste). Jeder Container wird bis zur
Abholung oder bis zum Ende des Beobachtungsfensters beobachtet; die Zensierzeit ist gleichverteilt zwischen 0 und dem Fenster, unabhängig von allem anderen.
Es laufen **{C.REPS} Läufe** mit festen Zufalls-Seeds; jeder zieht $n$ Container neu und schätzt das Merkmal zweimal: mit Cox und als Steigung der Geraden
$\\log(\\text{{beobachtete Zeit}}) \\sim x$.
"""
)
cn, cw = st.columns(2)
n = cn.slider("Container n je Lauf", *bounds("n_slider"), step=C.N_STEP, key="n_slider")
window = cw.select_slider("Beobachtungsfenster (Tage)", options=list(C.WINDOW_OPTIONS), key="window_select",
                          help="Obergrenze der gleichverteilten Zensierzeit. Je kürzer das Fenster, desto mehr Container sind zensiert; 1000 Tage heißt praktisch keine Zensierung.")

sync_query_params({
    "n_slider": int(n), "window_select": int(window), "beta_slider": float(beta), "shape_slider": float(shape), "seed_input": int(seed),
    "pattern_select": pattern, "probe_slider": float(probe),
})
world = World(int(n), int(window), float(beta), float(shape), int(seed))
with st.spinner(f"Rechne {C.REPS} feste Läufe (die Rechenzeit wächst mit n) …"):
    study, summary = _study(world)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Zensiert (Mittel)", ev.pct(summary.censored))
k2.metric("Cox β̂ (Mittel)", ev.num(summary.cox_mean, 3), delta=f"Wahrheit {ev.num(summary.truth_cox, 2)}", delta_color="off")
k3.metric("Regression, alle", ev.num(summary.ols_all_mean, 3), delta=f"Wahrheit {ev.num(summary.truth_ols, 3)}", delta_color="off")
k4.metric("Regression, nur Ereignisse", ev.num(summary.ols_events_mean, 3), delta=f"Wahrheit {ev.num(summary.truth_ols, 3)}", delta_color="off")
if summary.n_ok:
    st.plotly_chart(V.build_estimators(summary), width="stretch", key="estimators_chart")
    st.caption(
        "Fehlerbalken: ± 1 Standardfehler des Mittels über die Läufe. Die Wahrheit der Regression ist −β/k (bei Daten ohne Zensierung); sie braucht zusätzlich die Form k, "
        "die Cox-Regression nicht."
    )
kind, text = ev.verdict_study(summary, world)
getattr(st, kind)(text)
note = ev.failure_text(summary)
if note:
    st.warning(note)

st.markdown("**Wie sicher ist das Cox-β̂? Das Intervall β̂ ± 1,96·SE**")
if summary.n_ok:
    st.plotly_chart(V.build_intervals(study, world), width="stretch", key="intervals_chart")
    st.caption("Je Punkt ein Lauf; rot: das Intervall enthält die Wahrheit nicht.")
st.info(ev.coverage_text(summary))

st.markdown("---")

# --- Experiment 3: Grundrate nachträglich ------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Grundrate unbekannt, Kurven nachträglich")
st.markdown(
    f"""
Aus $\\hat\\beta$ allein ergibt sich keine Überlebenskurve; es fehlt die Grundrate. Der **Breslow-Schätzer** liefert sie nachträglich als Treppe aus den beobachteten Abholzeiten.
Hier ein Datensatz (Lauf 1 der Welt oben, $n$ = {world.n}): die Container werden nach dem Merkmal in drei etwa gleich große Gruppen geteilt. **Durchgezogen:** Kaplan–Meier je Gruppe
(ohne Modell). **Gestrichelt:** die Cox-Kurve der Gruppe, das Mittel der Kurven $e^{{-\\hat H_0(t)\\,e^{{\\hat\\beta x_i}}}}$ über ihre Mitglieder.
"""
)
curves = _curves(world)
if curves.fit.ok:
    tau = curves.comparison.tau if curves.comparison is not None else 0.0
    st.plotly_chart(V.build_group_curves(curves.sample, world, curves.fit, tau), width="stretch", key="curves_chart")
    st.caption(f"Cox β̂ in diesem Datensatz: {ev.num(curves.fit.beta, 3)} (Standardfehler {ev.num(curves.fit.se, 3)}; Wahrheit {ev.num(world.beta, 2)}).")
kind, text = ev.verdict_curves(curves)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was gilt, wenn sie verletzt ist |
|---|---|
| **Zensierung unabhängig vom Ereignis** | Nicht gemessen: Die Welt dieser Demo erzeugt unabhängige Zensierung, deshalb sagt keine Zahl hier etwas über abhängige Zensierung. |
| **Proportionale Raten** (der Faktor $e^{\\beta}$ bleibt über die Zeit gleich) | Nicht gemessen: Die Welt erfüllt sie. Ob echte Daten sie erfüllen, zeigt diese Demo nicht. |
| **Ein Merkmal, linear im Logarithmus der Rate** | Nicht gemessen: kein zweites Merkmal, keine nichtlineare Wirkung. |
| **Genug Ereignisse und Variation im Merkmal** | Gemessen: Bei kleinem $n$ (Regler in Experiment 2) hat ein Teil der Läufe keinen endlichen Schätzer; die App meldet den Grund (Trennung, Information 0, ohne Ereignis). |
| **Gleichstände der Zeiten** | Der Kern behandelt sie nach Breslow. Ob Efron genauer ist, ist hier nicht gemessen; die Demo nennt dazu keine Zahl. |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** *Gegeben:* je Container die beobachtete Zeit $t_i$, ob die Abholung beobachtet wurde ($\delta_i = 1$) oder der Container zensiert ist ($\delta_i = 0$), und das Merkmal $x_i$.
*Gesucht:* der Koeffizient $\beta$ und daraus das Hazard-Verhältnis $e^{\beta}$ je Einheit des Merkmals.

| Symbol | Bedeutung |
|---|---|
| $t_i,\ \delta_i,\ x_i$ | Beobachtungszeit, Ereignis (1) oder zensiert (0), Merkmal des Containers $i$ |
| $h(t \mid x)$ | Rate (Hazard) der Abholung zur Zeit $t$ bei Merkmal $x$ |
| $h_0(t)$ | Grundrate: die Rate bei $x = 0$, unbekannt und ohne feste Form |
| $\beta$ | Koeffizient; $e^{\beta}$ ist das Verhältnis der Raten bei $x+1$ gegen $x$ |
| $t_{(1)} < \dots < t_{(K)}$ | die verschiedenen Zeiten mit Abholung |
| $R_k,\ D_k,\ d_k$ | Risikomenge zur Zeit $t_{(k)}$ (alle mit $t_i \ge t_{(k)}$), Menge der dann Abgeholten, ihre Zahl |
| $m_k(\beta)$ | gewichtetes Mittel der Merkmale in $R_k$ mit Gewichten $e^{\beta x_j}$ |
| $U(\beta),\ I(\beta)$ | Score (erste Ableitung) und Information (negative zweite Ableitung) der partiellen Log-Likelihood |
| $\hat H_0(t)$ | geschätzte kumulierte Grundrate (Breslow) |
| $k,\ \lambda$ | Form und Skala der Weibull-Grundrate in der Simulation |

**Modell.**

$$h(t \mid x) = h_0(t)\, e^{\beta x}$$

**Partielle Likelihood (Breslow bei Gleichständen).** Die Grundrate $h_0$ kürzt sich heraus:

$$\ell(\beta) = \sum_{k=1}^{K} \Big[ \beta \sum_{i \in D_k} x_i \;-\; d_k \log \sum_{j \in R_k} e^{\beta x_j} \Big]$$

**Score und Information.**

$$U(\beta) = \sum_{k=1}^{K} \sum_{i \in D_k} \big( x_i - m_k(\beta) \big), \qquad m_k(\beta) = \frac{\sum_{j \in R_k} x_j\, e^{\beta x_j}}{\sum_{j \in R_k} e^{\beta x_j}}$$

$$I(\beta) = \sum_{k=1}^{K} d_k \Big[ \frac{\sum_{j \in R_k} x_j^2\, e^{\beta x_j}}{\sum_{j \in R_k} e^{\beta x_j}} - m_k(\beta)^2 \Big]$$

**Newton-Schritt und Standardfehler.**

$$\beta_{j+1} = \beta_j + \frac{U(\beta_j)}{I(\beta_j)}, \qquad \operatorname{SE}(\hat\beta) = \frac{1}{\sqrt{I(\hat\beta)}}$$

**Grundrate nach Breslow und Kurve einer Gruppe $G$.**

$$\hat H_0(t) = \sum_{t_{(k)} \le t} \frac{d_k}{\sum_{j \in R_k} e^{\hat\beta x_j}}, \qquad \hat S_G(t) = \frac{1}{|G|} \sum_{i \in G} \exp\!\big( -\hat H_0(t)\, e^{\hat\beta x_i} \big)$$

**Simulierte Welt.** Mit $E \sim \mathrm{Exponential}(1)$ ist die Abholzeit
$$T = \lambda \Big( \frac{E}{e^{\beta x}} \Big)^{1/k}$$
mit $S(t \mid x) = \exp\!\big( -(t/\lambda)^k e^{\beta x} \big)$. Ohne Zensierung gilt $\log T = \log\lambda + \tfrac{1}{k}\log E - \tfrac{\beta}{k}\, x$; die Steigung der Regression von $\log T$ auf $x$ ist daher $-\beta/k$
(das ist die Wahrheit, gegen die Experiment 2 die Regression hält).

**Voraussetzungen und Konvention.** Zensierung unabhängig vom Ereignis; Risikomenge $R_k = \{ i : t_i \ge t_{(k)} \}$, bei gleicher Zeit zählt das Ereignis vor der Zensierung. Bei perfekter Trennung
(der Abgeholte hat in jeder Risikomenge den größten Merkmalswert, oder in jeder Risikomenge den kleinsten) wächst $\ell(\beta)$ monoton und hat kein Maximum.

**Literatur** (beide aus dem Gedächtnis genannt, die Originale sind nicht eingesehen): D. R. Cox (1972), Regression Models and Life-Tables, Journal of the Royal Statistical Society B 34(2), 187–220;
N. E. Breslow (1974), Covariance Analysis of Censored Survival Data, Biometrics 30(1), 89–99. Die Formeln oben sind gegen unabhängige Rechnungen geprüft (`tests/test_oracle_cox.py`).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Überlebenszeit: zensierte Zeiten lesen](https://sebastianhanisch.net/konzepte-ueberlebenszeit.html)."
)
