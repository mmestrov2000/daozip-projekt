"""Smoke testovi za konfiguracijski sloj (F0.1/F0.2)."""

from src.config import CONFIG, load_config
from src import utils


def test_config_loads():
    """config.yaml se učitava kao rječnik s očekivanim ključevima."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert cfg["project_start"] == "2000-01"
    assert cfg["project_end"] == "2025-12"


def test_constants_bound_to_config():
    """Konstante u src.utils vežu se na vrijednosti iz configa."""
    assert utils.PROJECT_START == CONFIG["project_start"] == "2000-01"
    assert utils.PROJECT_END == CONFIG["project_end"] == "2025-12"
    assert utils.W_MAX == CONFIG["w_max"]
    assert utils.GROUP_CAP == CONFIG["group_cap"]
    assert utils.TRAIN_LOOKBACK_MONTHS == CONFIG["train_lookback_months"]
    assert utils.TEST_HORIZON_MONTHS == CONFIG["test_horizon_months"]
    assert utils.REFIT_STEP_MONTHS == CONFIG["refit_step_months"]


def test_epsilon_grid_has_four_members():
    """ε-mreža ima točno 4 člana."""
    assert len(CONFIG["epsilon_grid"]) == 4
    assert utils.EPSILON_GRID == CONFIG["epsilon_grid"]


def test_seed_is_int():
    """Sjeme je cijeli broj."""
    assert isinstance(CONFIG["random_seed"], int)
    assert isinstance(utils.RANDOM_SEED, int)


def test_mcs_params():
    """MCS parametri postoje i vežu se na konstante."""
    assert utils.MCS_ALPHA == CONFIG["mcs_alpha"] == 0.10
    assert utils.MCS_LOSS == CONFIG["mcs_loss"]
