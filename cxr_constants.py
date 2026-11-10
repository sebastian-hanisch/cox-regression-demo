"""Konstanten: Regler-Grenzen, feste Annahmen, Farben, Presets und der Annahmen-Text (App und README)."""

# --- Welt (Weibull-Grundrate, ein Merkmal, Zensierung durch ein Beobachtungsfenster) -----------------------------------------------------------
WEIBULL_SCALE = 8.0                       # Skala λ der Weibull-Grundrate (Tage); Median der Verweilzeit bei x = 0: λ·(ln 2)^(1/k)

N_MIN, N_MAX, DEFAULT_N = 10, 600, 300    # Komplexität am Maximum: je Lauf O(Ereignisse · n) im Newton-Schritt, mal REPS Läufe
N_STEP = 10
BETA_MIN, BETA_MAX, DEFAULT_BETA, BETA_STEP = 0.0, 1.5, 0.7, 0.1
SHAPE_MIN, SHAPE_MAX, DEFAULT_SHAPE, SHAPE_STEP = 0.5, 3.0, 1.5, 0.1
WINDOW_OPTIONS = (4, 6, 8, 12, 20, 40, 100, 1000)   # Obergrenze der Zensierzeit in Tagen; 1000 = praktisch keine Zensierung
DEFAULT_WINDOW = 12
SEED_MAX, DEFAULT_SEED = 999999, 35

REPS = 200                                # Läufe je Experiment (feste Seeds), in der App und in tools/measure_cox.py
Z95 = 1.959963984540054                   # Normalquantil für das zweiseitige 95-%-Intervall

# --- Experiment 1: Handbeispiel ----------------------------------------------------------------------------------------------------------------
PROBE_MIN, PROBE_MAX, DEFAULT_PROBE, PROBE_STEP = -1.0, 3.0, 0.0, 0.05   # β zum Ausprobieren
BETA_GRID = tuple(i / 20 for i in range(-20, 81))                         # Kurve der partiellen Log-Likelihood: β von −1 bis 4

# --- Experiment 3: Gruppenkurven ---------------------------------------------------------------------------------------------------------------
GROUP_CUTS = (-0.4307, 0.4307)            # Terzile der Standardnormalverteilung: drei etwa gleich große Gruppen nach dem Merkmal
MIN_AT_RISK = 10                          # Vergleichsbereich bis zur Zeit, zu der in jeder Gruppe noch so viele im Risiko sind

COLOR_COX = "#2F6B65"
COLOR_OLS = "#c0504d"
COLOR_TRUTH = "#7f7f7f"
COLOR_LINE = "#4c78a8"
GROUP_COLORS = ("#4c78a8", "#59a14f", "#e15759")

# --- Presets ------------------------------------------------------------------------------------------------------------------------------------
# Schlüssel: n, window, beta, shape, seed (Welt) und pattern (Experiment 1); fehlt ein Schlüssel, gilt der Standardwert.
PRESETS = {
    "Messung V6 (Standard)": {"n": 300, "window": 12, "beta": 0.7, "shape": 1.5, "seed": DEFAULT_SEED},
    "Kaum zensiert": {"n": 300, "window": 1000, "beta": 0.7, "shape": 1.5, "seed": DEFAULT_SEED},
    "Stark zensiert": {"n": 300, "window": 4, "beta": 0.7, "shape": 1.5, "seed": DEFAULT_SEED},
    "Kleine Stichprobe (n = 50)": {"n": 50, "window": 12, "beta": 0.7, "shape": 1.5, "seed": DEFAULT_SEED},
    "Ohne Wirkung (β = 0)": {"n": 300, "window": 12, "beta": 0.0, "shape": 1.5, "seed": DEFAULT_SEED},
    "Perfekte Trennung (Handbeispiel)": {"pattern": "Perfekte Trennung: x = 1, 1, 0, 0"},
}
PRESET_HELP = {
    "Messung V6 (Standard)": "Weibull-Form 1,5, ein Standardnormal-Merkmal, β = 0,7, 300 Container, Beobachtungsfenster 12 Tage; das ist die Welt der Messungen dieser Demo.",
    "Kaum zensiert": "Beobachtungsfenster 1000 Tage: fast jeder Container wird bis zur Abholung beobachtet. Die Regression auf die Zeit hat dann kein Zensierungsproblem.",
    "Stark zensiert": "Beobachtungsfenster 4 Tage: die meisten Container sind bei Fensterende noch nicht abgeholt.",
    "Kleine Stichprobe (n = 50)": "50 Container: das Konfidenzintervall ist breit, und in einzelnen Läufen gibt es keinen Schätzer.",
    "Ohne Wirkung (β = 0)": "Das Merkmal ändert die Rate nicht: beide Schätzer müssen um 0 streuen.",
    "Perfekte Trennung (Handbeispiel)": "Experiment 1 mit x = 1, 1, 0, 0: alle Container mit Merkmal 1 werden zuerst abgeholt, und es gibt keinen endlichen Schätzer.",
}

# --- Annahmen, Grenzen und bekannte Probleme (derselbe Text steht in der README; ein Test vergleicht beide) -------------------------------------
ASSUMPTIONS_MARKDOWN = """\
**Annahmen**
- Die Zensierung ist unabhängig vom Ereignis: Ein Container, der zensiert wird, hat dieselbe Abholrate wie die, die noch im Risiko bleiben (hier erzeugt durch ein Beobachtungsfenster mit gleichverteilter Zensierzeit, unabhängig von Merkmal und Abholzeit).
- Proportionale Raten: Das Merkmal vervielfacht die Rate mit einem Faktor exp(β), der über die Zeit gleich bleibt. Die Welt dieser Demo erfüllt das; ob echte Daten es tun, prüft die Demo nicht.
- Ein Merkmal, logarithmisch linear in der Rate; die Grundform der Rate bleibt unbestimmt (sie wird nicht geschätzt, sondern nur über die Risikomengen herausgekürzt).
- Gleichstände der Zeiten werden nach Breslow behandelt (Risikomenge bleibt für alle Ereignisse derselben Zeit gleich).

**Grenzen**
- Alle Zahlen stammen aus einer simulierten Welt (Weibull-Grundrate, ein normalverteiltes Merkmal, Zensierung durch ein Beobachtungsfenster); andere Welten verändern die Größen, nicht unbedingt die Richtung.
- Die Überdeckung des Intervalls β̂ ± 1,96·SE ist nur für das richtige Modell gemessen, nicht bei verletzter Proportionalität.
- Es gibt keine Auswahl unter mehreren Merkmalen; ein Merkmal live.
- Gleichstände: Breslow gegen Efron ist nicht gemessen, die Demo macht dazu keine Zahlenaussage.

**Bekannte Probleme und Unsicherheiten**
- Die Läufe in Experiment 2 und 3 sind Stichproben mit festen Zufalls-Seeds: Anteile wie „in 94 von 100 Läufen“ schwanken um einige Prozentpunkte, wenn man den Zufalls-Seed ändert.
- Bei perfekter Trennung, konstantem Merkmal oder ohne Ereignis gibt es keinen endlichen Schätzer; die App meldet das, statt eine Zahl zu zeigen.
- Die Seite ist im Browser nur am Rechner geprüft (Seitenaufruf, Regler über AppTest), nicht auf Mobilgeräten.
"""
