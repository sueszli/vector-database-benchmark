import logging
import pytest
from tests.conftest import TrackedContainer
from tests.package_helper import CondaPackageHelper
LOGGER = logging.getLogger(__name__)

@pytest.mark.info
def test_outdated_packages(container: TrackedContainer, requested_only: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Getting the list of updatable packages'
    LOGGER.info(f'Checking outdated packages in {container.image_name} ...')
    pkg_helper = CondaPackageHelper(container)
    pkg_helper.check_updatable_packages(requested_only)
    LOGGER.info(pkg_helper.get_outdated_summary(requested_only))
    LOGGER.info(f'\n{pkg_helper.get_outdated_table()}\n')