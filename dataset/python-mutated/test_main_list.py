import json
from pathlib import Path
import pytest
from pytest_mock import MockerFixture
from conda.exceptions import EnvironmentLocationNotFound
from conda.testing import CondaCLIFixture, TmpEnvFixture

@pytest.fixture
def tmp_envs_dirs(mocker: MockerFixture, tmp_path: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    mocker.patch('conda.base.context.mockable_context_envs_dirs', return_value=(str(tmp_path),))
    return tmp_path

def test_list(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        while True:
            i = 10
    pkg = 'ca-certificates'
    with tmp_env(pkg) as prefix:
        (stdout, _, _) = conda_cli('list', '--prefix', prefix, '--json')
        assert any((item['name'] == pkg for item in json.loads(stdout)))

def test_list_reverse(tmp_env: TmpEnvFixture, conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    pkg = 'curl'
    with tmp_env(pkg) as prefix:
        (stdout, _, _) = conda_cli('list', '--prefix', prefix, '--json')
        names = [item['name'] for item in json.loads(stdout)]
        assert names == sorted(names)
        (stdout, _, _) = conda_cli('list', '--prefix', prefix, '--reverse', '--json')
        names = [item['name'] for item in json.loads(stdout)]
        assert names == sorted(names, reverse=True)

def test_list_json(tmp_envs_dirs: Path, conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    (stdout, _, _) = conda_cli('list', '--json')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, list)
    with pytest.raises(EnvironmentLocationNotFound):
        conda_cli('list', '--name', 'nonexistent', '--json')

def test_list_revisions(tmp_envs_dirs: Path, conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    (stdout, _, _) = conda_cli('list', '--revisions', '--json')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, list) or (isinstance(parsed, dict) and 'error' in parsed)
    with pytest.raises(EnvironmentLocationNotFound):
        conda_cli('list', '--name', 'nonexistent', '--revisions', '--json')

def test_list_package(tmp_envs_dirs: Path, conda_cli: CondaCLIFixture):
    if False:
        i = 10
        return i + 15
    (stdout, _, _) = conda_cli('list', 'ipython', '--json')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, list)