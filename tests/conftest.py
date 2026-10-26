import os

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    """Isolate tests from the developer's LKA_* variables and any real .env file."""
    for name in list(os.environ):
        if name.startswith("LKA_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
