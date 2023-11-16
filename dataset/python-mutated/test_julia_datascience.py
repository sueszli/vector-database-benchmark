from tests.conftest import TrackedContainer
from tests.run_command import run_command

def test_julia(container: TrackedContainer) -> None:
    if False:
        i = 10
        return i + 15
    run_command(container, 'julia --version')