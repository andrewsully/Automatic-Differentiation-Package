import pytest

from autodiff_team29 import Node


@pytest.fixture(autouse=True, scope="function")
def teardown_module():
    """
    Once nodes are created, they will persist in the registry throughout the duration of the programs' execution, unless
    the registry is cleared. To prevent precomputed nodes persisting between test, we can clear the registry before and
    after each test unit test runs. We will also make sure that overwrite mode is off by default

    """
    Node.clear_node_registry()
    Node.set_overwrite_mode(False)
    yield
    Node.clear_node_registry()
    Node.set_overwrite_mode(False)
