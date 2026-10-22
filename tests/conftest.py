"""Run the tests from output/logs: biobb writes its log_*.out files and sandboxes to the
working directory, which would otherwise be the repository root."""
import os
from pathlib import Path

import pytest

LOG_DIR = Path(__file__).resolve().parents[1] / 'output' / 'logs'


@pytest.fixture(scope='session', autouse=True)
def biobb_logs_in_output():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(LOG_DIR)
    yield
    os.chdir(cwd)
