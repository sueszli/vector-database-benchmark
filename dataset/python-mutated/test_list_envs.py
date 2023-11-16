from __future__ import annotations
from typing import TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    from tox.pytest import ToxProject, ToxProjectCreator

@pytest.fixture()
def project(tox_project: ToxProjectCreator) -> ToxProject:
    if False:
        for i in range(10):
            print('nop')
    ini = '\n    [tox]\n    env_list=py32,py31,py\n    [testenv]\n    package = wheel\n    wheel_build_env = pkg\n    description = with {basepython}\n    deps = pypy:\n    [testenv:py]\n    basepython=py32,py31\n    [testenv:fix]\n    description = fix it\n    [testenv:pkg]\n    '
    return tox_project({'tox.ini': ini})

def test_list_env(project: ToxProject) -> None:
    if False:
        return 10
    outcome = project.run('l')
    outcome.assert_success()
    expected = '\n    default environments:\n    py32 -> with py32\n    py31 -> with py31\n    py   -> with py32 py31\n\n    additional environments:\n    fix  -> fix it\n    pypy -> with pypy\n    '
    outcome.assert_out_err(expected, '')

def test_list_env_default(project: ToxProject) -> None:
    if False:
        while True:
            i = 10
    outcome = project.run('l', '-d')
    outcome.assert_success()
    expected = '\n    default environments:\n    py32 -> with py32\n    py31 -> with py31\n    py   -> with py32 py31\n    '
    outcome.assert_out_err(expected, '')

def test_list_env_quiet(project: ToxProject) -> None:
    if False:
        return 10
    outcome = project.run('l', '--no-desc')
    outcome.assert_success()
    expected = '\n    py32\n    py31\n    py\n    fix\n    pypy\n    '
    outcome.assert_out_err(expected, '')

def test_list_env_quiet_default(project: ToxProject) -> None:
    if False:
        while True:
            i = 10
    outcome = project.run('l', '--no-desc', '-d')
    outcome.assert_success()
    expected = '\n    py32\n    py31\n    py\n    '
    outcome.assert_out_err(expected, '')

def test_list_env_package_env_before_run(tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    ini = '\n        [testenv:pkg]\n        [testenv:run]\n        package = wheel\n        wheel_build_env = pkg\n    '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l')
    outcome.assert_success()
    expected = '\n    default environments:\n    py  -> [no description]\n\n    additional environments:\n    run -> [no description]\n    '
    outcome.assert_out_err(expected, '')

def test_list_env_package_self(tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    ini = '\n        [tox]\n        env_list = pkg\n        [testenv:pkg]\n        package = wheel\n        wheel_build_env = pkg\n    '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l')
    outcome.assert_failed()
    assert outcome.out.splitlines() == ['ROOT: HandledError| pkg cannot self-package']

def test_list_envs_help(tox_project: ToxProjectCreator) -> None:
    if False:
        print('Hello World!')
    outcome = tox_project({'tox.ini': ''}).run('l', '-h')
    outcome.assert_success()