"""Zajednički pomoćnici za crtanje koji se ponovno koriste kroz notebookove.

Prije su bili iznova definirani izravno u svakom notebooku; njihovo sabiranje ovdje
daje projektu jedinstven, uvozljiv vizualizacijski modul (datoteka je nekoć bila
prazna zaglavna datoteka). Notebookovi i dalje ostaju pokretljivi samostalno, ali nove analize
(notebookovi 07-08) i svako ad hoc izvještavanje mogu ih uvesti izravno.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

# Kanonska paleta / prikazni nazivi portfelja (usklađeno s notebookovima 01-06).
PALETTE = {
    "equal_weight": "#888888",
    "min_var": "#1f77b4",
    "min_var_sector": "#2ca02c",
    "min_var_corr_cluster": "#9467bd",
    "min_var_factor_cluster": "#d62728",
    "factor_neutral_e0": "#ff7f0e",
}
DISPLAY = {
    "equal_weight": "Jednake težine",
    "min_var": "Minimalna varijanca",
    "min_var_sector": "Ograničenje sektora",
    "min_var_corr_cluster": "Ograničenje korel. klastera",
    "min_var_factor_cluster": "Ograničenje fakt. klastera",
    "factor_neutral_e0": "Faktorski neutralan (ε=0)",
}


def radar_ax(ax, values: Sequence[float], labels: Sequence[str], color: str = "C0",
             fill_alpha: float = 0.2, label: str | None = None) -> None:
    """Nacrtaj zatvoreni radarski/paukov poligon za ``values`` na polarnoj osi."""
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = list(values) + [values[0]]
    angles_closed = angles + [angles[0]]
    ax.plot(angles_closed, values_closed, color=color, linewidth=1.6, label=label)
    ax.fill(angles_closed, values_closed, alpha=fill_alpha, color=color)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels([])


def cluster_weights(weights: pd.DataFrame, ticker_to_cluster: dict) -> pd.Series:
    """Zbroji težine portfelja unutar svakog faktorskog klastera.

    ``weights`` mora imati stupce ``ticker`` i ``weight``; ``ticker_to_cluster``
    preslikava svaku oznaku dionice na njezin (cjelobrojni) id klastera. Oznake bez klastera
    se izbacuju.
    """
    frame = weights.copy()
    frame["cluster"] = frame["ticker"].map(ticker_to_cluster)
    frame = frame.dropna(subset=["cluster"])
    frame["cluster"] = frame["cluster"].astype(int)
    return frame.groupby("cluster")["weight"].sum()


def save_fig(fig, path, dpi: int = 150) -> None:
    """Spremi figuru tijesno obrezanu uz dosljedan DPI."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


__all__ = ["PALETTE", "DISPLAY", "radar_ax", "cluster_weights", "save_fig"]
