"""Testovi stabilnosti klastera kroz uzastopne prozore (F3.4)."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import ari_between_consecutive_windows


def _cluster_table(window_to_labels: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = [
        {"train_window": window, "ticker": ticker, "cluster": label}
        for window, labels in window_to_labels.items()
        for ticker, label in labels.items()
    ]
    return pd.DataFrame(rows)


def test_identical_labels_give_ari_one():
    """Identične oznake u oba prozora → ARI = 1."""
    table = _cluster_table(
        {
            "2013-12": {"A": 1, "B": 1, "C": 2, "D": 2},
            "2014-12": {"A": 1, "B": 1, "C": 2, "D": 2},
        }
    )
    result = ari_between_consecutive_windows(table, "cluster")
    assert len(result) == 1
    assert result.loc[0, "ari"] == pytest.approx(1.0)
    assert result.loc[0, "n_common"] == 4


def test_relabeled_partition_still_ari_one():
    """ARI je invarijantan na permutaciju oznaka (ista particija → ARI = 1)."""
    table = _cluster_table(
        {
            "2013-12": {"A": 1, "B": 1, "C": 2, "D": 2},
            "2014-12": {"A": 7, "B": 7, "C": 3, "D": 3},
        }
    )
    result = ari_between_consecutive_windows(table, "cluster")
    assert result.loc[0, "ari"] == pytest.approx(1.0)


def test_different_partition_below_one():
    """Drukčija particija → ARI < 1."""
    table = _cluster_table(
        {
            "2013-12": {"A": 1, "B": 1, "C": 2, "D": 2},
            "2014-12": {"A": 1, "B": 2, "C": 1, "D": 2},
        }
    )
    result = ari_between_consecutive_windows(table, "cluster")
    assert result.loc[0, "ari"] < 1.0


def test_row_count_and_intersection():
    """Jedan redak po prijelazu; ARI se računa na presjeku dionica."""
    table = _cluster_table(
        {
            "2013-12": {"A": 1, "B": 1, "C": 2, "D": 2},
            "2014-12": {"A": 1, "B": 1, "C": 2, "E": 3},  # D izašao, E ušao
            "2015-12": {"A": 1, "B": 2, "C": 2, "E": 3},
        }
    )
    result = ari_between_consecutive_windows(table, "cluster")
    assert len(result) == 2  # prijelazi između tri prozora
    # presjek prvog prijelaza je {A, B, C}
    assert result.loc[0, "n_common"] == 3
    assert list(result["from_window"]) == ["2013-12", "2014-12"]
    assert list(result["to_window"]) == ["2014-12", "2015-12"]


def test_missing_columns_raise():
    table = pd.DataFrame({"train_window": ["2013-12"], "ticker": ["A"]})
    with pytest.raises(ValueError):
        ari_between_consecutive_windows(table, "cluster")
