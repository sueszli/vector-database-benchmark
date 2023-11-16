import pytest
from conda.testing import TmpEnvFixture

@pytest.mark.integration
def test_link_order_post_link_actions(test_recipes_channel: None, tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    with tmp_env('c_post_link_package', '--use-local'):
        pass

@pytest.mark.integration
def test_link_order_post_link_depend(test_recipes_channel: None, tmp_env: TmpEnvFixture):
    if False:
        i = 10
        return i + 15
    with tmp_env('e_post_link_package', '--use-local'):
        pass