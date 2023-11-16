import json
from pathlib import Path
from typing import Iterable
import pytest
from conda.base.context import context
from conda.testing import CondaCLIFixture, PathFactoryFixture

@pytest.mark.parametrize('args', [pytest.param(('--get',), id='get'), pytest.param(('--get', 'channels'), id='key'), pytest.param(('--get', 'use_pip'), id='unknown')])
def test_config_get_user(conda_cli: CondaCLIFixture, args: Iterable[str]):
    if False:
        return 10
    (stdout, _, _) = conda_cli('config', '--json', *args)
    parsed = json.loads(stdout.strip())
    assert 'get' in parsed
    assert 'rc_path' in parsed
    assert parsed['success']
    assert 'warnings' in parsed

@pytest.mark.skipif(not context.root_writable, reason='not root writable')
@pytest.mark.parametrize('args', [pytest.param(('--get',), id='get'), pytest.param(('--get', 'channels'), id='key'), pytest.param(('--get', 'use_pip'), id='unknown')])
def test_config_get_system(conda_cli: CondaCLIFixture, args: Iterable[str]):
    if False:
        while True:
            i = 10
    (stdout, _, _) = conda_cli('config', '--json', *args, '--system')
    parsed = json.loads(stdout.strip())
    assert 'get' in parsed
    assert 'rc_path' in parsed
    assert parsed['success']
    assert 'warnings' in parsed

@pytest.mark.parametrize('args', [pytest.param(('--get',), id='get'), pytest.param(('--get', 'channels'), id='key'), pytest.param(('--get', 'use_pip'), id='unknown')])
def test_config_get_missing(conda_cli: CondaCLIFixture, args: Iterable[str], path_factory: PathFactoryFixture):
    if False:
        i = 10
        return i + 15
    path = path_factory()
    (stdout, _, _) = conda_cli('config', '--json', *args, '--file', path)
    parsed = json.loads(stdout.strip())
    assert 'get' in parsed
    assert Path(parsed['rc_path']) == path
    assert parsed['success']
    assert 'warnings' in parsed

def test_config_show_sources_json(conda_cli: CondaCLIFixture):
    if False:
        return 10
    (stdout, stderr, err) = conda_cli('config', '--show-sources', '--json')
    parsed = json.loads(stdout.strip())
    assert 'error' not in parsed
    assert not stderr
    assert not err