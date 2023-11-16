import pytest
from tests.conftest import TrackedContainer
from tests.run_command import run_command

@pytest.mark.parametrize('package_manager_command', ['apt --version', 'conda --version', 'mamba --version', 'pip --version'])
def test_package_manager(container: TrackedContainer, package_manager_command: str) -> None:
    if False:
        return 10
    'Test that package managers are installed and run.'
    run_command(container, package_manager_command)