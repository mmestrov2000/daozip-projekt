"""Učitavanje konfiguracije projekta iz config.yaml.

Jedino mjesto koje čita YAML; konstante u src/utils.py vežu se na vrijednosti
koje vraća load_config(). Parametri se nikad ne hardkodiraju drugdje.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Učitaj i vrati konfiguraciju projekta kao rječnik."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Učitano jednom pri uvozu modula; svi ostali moduli koriste ovaj rječnik.
CONFIG: dict[str, Any] = load_config()
