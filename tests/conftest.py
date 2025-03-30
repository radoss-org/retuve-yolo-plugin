# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import warnings

import pytest
from retuve.defaults.hip_configs import default_US

default_US.register("ultrasound", live=True)


# Suppress specific DeprecationWarnings during tests
@pytest.fixture(autouse=True)
def suppress_warnings():
    warnings.filterwarnings("ignore")
