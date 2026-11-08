import yaml
import pytest
from pathlib import Path
from nse_ai.data import make_synthetic_nse
from nse_ai.features import build_panel

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def cfg():
    c = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text())
    c["learning"]["population"] = 30       # faster tests
    c["learning"]["generations"] = 4
    return c


@pytest.fixture(scope="session")
def md():
    return make_synthetic_nse(n_stocks=30, years=9, seed=2, signal=True, signal_strength=0.35)


@pytest.fixture(scope="session")
def panel(md, cfg):
    return build_panel(md, cfg)
