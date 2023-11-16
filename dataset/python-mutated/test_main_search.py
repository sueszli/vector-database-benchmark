import json
import re
import pytest
from conda.testing import CondaCLIFixture
pytestmark = [pytest.mark.integration]

@pytest.mark.flaky(reruns=5)
def test_search_0(conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    (stdout, stderr, err) = conda_cli('search', '*[build=py_3]', '--json', '--override-channels', '--channel', 'defaults')
    assert not stderr
    assert not err
    parsed = json.loads(stdout.strip())
    package_name = 'pydotplus'
    assert isinstance(parsed, dict)
    assert isinstance(parsed[package_name], list)
    assert isinstance(parsed[package_name][0], dict)
    assert {'build', 'channel', 'fn', 'version'} <= set(parsed[package_name][0])
    assert parsed[package_name][0]['build'] == 'py_3'

@pytest.mark.flaky(reruns=5)
def test_search_1(conda_cli: CondaCLIFixture):
    if False:
        i = 10
        return i + 15
    (stdout, stderr, err) = conda_cli('search', 'ipython', '--json', '--override-channels', '--channel', 'defaults')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, dict)
    assert not stderr
    assert not err

@pytest.mark.parametrize('package', [pytest.param('python', id='exact'), pytest.param('ython', id='wildcard')])
@pytest.mark.flaky(reruns=5)
def test_search_2(conda_cli: CondaCLIFixture, package: str):
    if False:
        while True:
            i = 10
    (stdout, stderr, err) = conda_cli('search', package, '--override-channels', '--channel', 'defaults')
    assert re.search('(python)\\s+(\\d+\\.\\d+\\.\\d+)\\s+(\\w+)\\s+(pkgs/main)', stdout)
    assert not stderr
    assert not err

@pytest.mark.flaky(reruns=5)
def test_search_3(conda_cli: CondaCLIFixture):
    if False:
        i = 10
        return i + 15
    (stdout, stderr, err) = conda_cli('search', '*/linux-64::nose==1.3.7[build=py37_2]', '--info', '--override-channels', '--channel', 'defaults')
    assert 'file name   : nose-1.3.7-py37_2' in stdout
    assert 'name        : nose' in stdout
    assert 'version     : 1.3.7' in stdout
    assert 'build       : py37_2' in stdout
    assert 'build number: 2' in stdout
    assert 'subdir      : linux-64' in stdout
    assert 'url         : https://repo.anaconda.com/pkgs/main/linux-64/nose-1.3.7-py37_2' in stdout
    assert not stderr
    assert not err

@pytest.mark.flaky(reruns=5)
def test_search_4(conda_cli: CondaCLIFixture):
    if False:
        print('Hello World!')
    (stdout, stderr, err) = conda_cli('search', '--json', '--override-channels', '--channel', 'defaults', '--use-index-cache', 'python')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, dict)
    assert not stderr
    assert not err

@pytest.mark.flaky(reruns=5)
def test_search_5(conda_cli: CondaCLIFixture):
    if False:
        for i in range(10):
            print('nop')
    (stdout, stderr, err) = conda_cli('search', '--platform', 'win-32', '--json', '--override-channels', '--channel', 'defaults', 'python')
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, dict)
    assert not stderr
    assert not err

def test_search_envs(conda_cli: CondaCLIFixture):
    if False:
        i = 10
        return i + 15
    (stdout, _, _) = conda_cli('search', '--envs', 'python')
    assert 'Searching environments' in stdout
    assert 'python' in stdout

def test_search_envs_info(conda_cli: CondaCLIFixture):
    if False:
        i = 10
        return i + 15
    (stdout, _, _) = conda_cli('search', '--envs', '--info', 'python')
    assert 'Searching environments' in stdout
    assert 'python' in stdout

def test_search_envs_json(conda_cli: CondaCLIFixture):
    if False:
        print('Hello World!')
    search_for = 'python'
    (stdout, _, _) = conda_cli('search', '--envs', '--json', search_for)
    assert 'Searching environments' not in stdout
    parsed = json.loads(stdout.strip())
    assert isinstance(parsed, list)
    assert len(parsed), 'empty search result'
    assert all((entry['package_records'][0]['name'] == search_for for entry in parsed))