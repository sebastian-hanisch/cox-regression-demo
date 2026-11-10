"""Gemeinsame Fixtures: das Vier-Container-Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture
def hand():
    """Vier Container mit Merkmal x = 1, 0, 1, 0, abgeholt zu den Zeiten 1, 2, 3, 4 (alle beobachtet).
    Handrechnung bei β = 0: Risikomengen {1,0,1,0}, {0,1,0}, {1,0}, {0}; Mittel 1/2, 1/3, 1/2, 0; Score 2/3; Information 13/18; β̂ = 0,9406."""
    return np.array([1.0, 2.0, 3.0, 4.0]), np.ones(4, dtype=bool), np.array([1.0, 0.0, 1.0, 0.0])
