import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run slow tests (build the full CLIP+DINOv2 backbone)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (needs the full backbone / a GPU)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
