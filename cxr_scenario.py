"""Szenario: Container mit einem Merkmal, Weibull-Grundrate und zufälliger Zensierung (Beobachtungsfenster).

Die Welt (DEMO-PLAYBOOK §1b: eine Regel je Funktion, Zufall nur über SplitMix64):
  * Merkmal x standardnormal,
  * Rate h(t | x) = h0(t) · exp(β·x) mit Weibull-Grundrate h0(t) = (k/λ)(t/λ)^(k−1),
    also T = λ · (E / exp(β·x))^(1/k) mit E ~ Exponential(1),
  * Zensierzeit C gleichverteilt auf (0, Fenster), unabhängig von T und x; beobachtet wird min(T, C), das Ereignis ist T ≤ C.
Dazu das Vier-Container-Handbeispiel (Experiment 1) mit den Mustern des Merkmals.
"""

from dataclasses import dataclass

import numpy as np

import cxr_constants as C
from cxr_core import SplitMix64


@dataclass(frozen=True)
class World:
    """Einstellungen der simulierten Welt; alles, was die Daten bestimmt, steht hier (gehört in jeden Cache-Schlüssel)."""

    n: int = C.DEFAULT_N
    window: float = C.DEFAULT_WINDOW
    beta: float = C.DEFAULT_BETA
    shape: float = C.DEFAULT_SHAPE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Sample:
    time: np.ndarray    # beobachtete Zeit min(T, C)
    event: np.ndarray   # True: Ereignis (Abholung), False: zensiert
    x: np.ndarray       # Merkmal
    latent: np.ndarray  # wahre Zeit T (nur der Simulation bekannt; Zensierte kennen sie in der Praxis nicht)


def run_seed(seed: int, run: int) -> int:
    """Seed des Laufs `run` aus dem Zufalls-Seed der Einstellung: eigener, fester Strom je Lauf."""
    return seed * 1_000_003 + run


def true_time(x, e, beta: float, shape: float, scale: float = C.WEIBULL_SCALE):
    """Wahre Zeit aus Merkmal x und Exponential-Zufall e: T = λ (E / exp(βx))^(1/k) (Inversion der Überlebensfunktion)."""
    return scale * (np.asarray(e) / np.exp(beta * np.asarray(x))) ** (1.0 / shape)


def simulate(world: World, run: int = 0) -> Sample:
    """Ein Datensatz der Welt. Zugstrom je Lauf: erst n Merkmale (normal), dann n Exponentialzufälle, dann n Zensierzeiten (gleichverteilt)."""
    rng = SplitMix64(run_seed(world.seed, run))
    x = rng.normal(world.n)
    e = rng.exponential(world.n)
    c = rng.uniform(world.n) * world.window
    t = true_time(x, e, world.beta, world.shape)
    obs = np.maximum(np.minimum(t, c), 1e-9)  # Zeiten müssen > 0 sein (der Kern prüft das)
    return Sample(time=obs, event=t <= c, x=x, latent=t)


def true_survival(t, x, beta: float, shape: float, scale: float = C.WEIBULL_SCALE):
    """Wahre Überlebensfunktion S(t | x) = exp(−(t/λ)^k · exp(βx))."""
    return np.exp(-((np.asarray(t, dtype=float) / scale) ** shape) * np.exp(beta * np.asarray(x, dtype=float)))


# ------------------------------------------------------------------------------------------------ Handbeispiel (Experiment 1)

HAND_TIMES = (1.0, 2.0, 3.0, 4.0)

# Muster des Merkmals der vier Container (Abholung nacheinander zu den Zeiten 1, 2, 3, 4)
PATTERNS = {
    "Handbeispiel: x = 1, 0, 1, 0": (1.0, 0.0, 1.0, 0.0),
    "Umgekehrt: x = 0, 1, 0, 1": (0.0, 1.0, 0.0, 1.0),
    "Perfekte Trennung: x = 1, 1, 0, 0": (1.0, 1.0, 0.0, 0.0),
    "Konstantes Merkmal: x = 1, 1, 1, 1": (1.0, 1.0, 1.0, 1.0),
}
DEFAULT_PATTERN = next(iter(PATTERNS))
