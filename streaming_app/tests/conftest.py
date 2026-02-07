"""
Shared fixtures for streaming_app tests.
Requires GPU + models loaded. Run with: uv run pytest streaming_app/tests/ -v
"""

import pytest
import yaml
import torch
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def config():
    config_path = Path("streaming_app/config/streaming_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)["streaming"]


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def loaded_models(config):
    """Load all models once for the entire test session."""
    from streaming_app.models.model_loader import model_loader
    models = model_loader.load_all_models(config)
    return models


@pytest.fixture(scope="session")
def dystream_model(loaded_models):
    return loaded_models["dystream"]


@pytest.fixture(scope="session")
def vis_models(loaded_models):
    return loaded_models["visualization"]


@pytest.fixture(scope="session")
def noise_scheduler(loaded_models):
    return loaded_models.get("noise_scheduler")


@pytest.fixture(scope="session")
def ema(loaded_models):
    return loaded_models.get("dystream_ema")
