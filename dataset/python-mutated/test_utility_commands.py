import re
import subprocess
import pytest
from tests.semgrep_runner import SEMGREP_BASE_COMMAND

@pytest.mark.kinda_slow
def test_version():
    if False:
        while True:
            i = 10
    result = subprocess.check_output(SEMGREP_BASE_COMMAND + ['--version', '--disable-version-check'], encoding='utf-8')
    assert re.match('\\d+\\.\\d+\\.\\d+', result)

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_dump_command_for_core():
    if False:
        for i in range(10):
            print('nop')
    semgrep_core_command = subprocess.check_output(SEMGREP_BASE_COMMAND + ['--config', 'tests/e2e/rules/eqeq-basic.yaml', 'tests/e2e/targets/basic', '-d'], encoding='utf-8')
    result = subprocess.run(semgrep_core_command, shell=True)
    assert result.returncode == 0

@pytest.mark.kinda_slow
@pytest.mark.osemfail
def test_dump_engine():
    if False:
        return 10
    result = subprocess.check_output(SEMGREP_BASE_COMMAND + ['--dump-engine-path'], encoding='utf-8')
    assert re.match('/[\\w/]+/semgrep-core', result)