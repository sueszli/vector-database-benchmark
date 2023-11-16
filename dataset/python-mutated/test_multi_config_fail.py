import pytest
from tests.fixtures import RunSemgrep

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_multi_config_fail(run_semgrep_in_tmp: RunSemgrep):
    if False:
        for i in range(10):
            print('nop')
    run_semgrep_in_tmp(['rules/multi_config_fail/error.yaml', 'rules/multi_config_fail/no_error.yaml'], target_name='basic/stupid.py', assert_exit_code=7)