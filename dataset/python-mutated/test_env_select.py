from __future__ import annotations
import sys
from typing import TYPE_CHECKING
import pytest
from tox.config.cli.parse import get_options
from tox.session.env_select import _DYNAMIC_ENV_FACTORS, CliEnv, EnvSelector
from tox.session.state import State
if TYPE_CHECKING:
    from tox.pytest import MonkeyPatch, ToxProjectCreator
CURRENT_PY_ENV = f'py{sys.version_info[0]}{sys.version_info[1]}'

def test_label_core_can_define(tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    ini = '\n        [tox]\n        labels =\n            test = py3{10,9}\n            static = flake8, type\n        '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l', '--no-desc')
    outcome.assert_success()
    outcome.assert_out_err('py\npy310\npy39\nflake8\ntype\n', '')

def test_label_core_select(tox_project: ToxProjectCreator) -> None:
    if False:
        return 10
    ini = '\n        [tox]\n        labels =\n            test = py3{10,9}\n            static = flake8, type\n        '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l', '--no-desc', '-m', 'test')
    outcome.assert_success()
    outcome.assert_out_err('py310\npy39\n', '')

def test_label_select_trait(tox_project: ToxProjectCreator) -> None:
    if False:
        return 10
    ini = '\n        [tox]\n        env_list = py310, py39, flake8, type\n        [testenv]\n        labels = test\n        [testenv:flake8]\n        labels = static\n        [testenv:type]\n        labels = static\n        '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l', '--no-desc', '-m', 'test')
    outcome.assert_success()
    outcome.assert_out_err('py310\npy39\n', '')

def test_label_core_and_trait(tox_project: ToxProjectCreator) -> None:
    if False:
        for i in range(10):
            print('nop')
    ini = '\n        [tox]\n        env_list = py310, py39, flake8, type\n        labels =\n            static = flake8, type\n        [testenv]\n        labels = test\n        '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l', '--no-desc', '-m', 'test', 'static')
    outcome.assert_success()
    outcome.assert_out_err('py310\npy39\nflake8\ntype\n', '')

@pytest.mark.parametrize(('selection_arguments', 'expect_envs'), [(('-f', 'cov', 'django20'), ('py310-django20-cov', 'py39-django20-cov')), (('-f', 'cov-django20'), ('py310-django20-cov', 'py39-django20-cov')), (('-f', 'py39', 'django20', '-f', 'py310', 'django21'), ('py310-django21-cov', 'py310-django21', 'py39-django20-cov', 'py39-django20'))])
def test_factor_select(tox_project: ToxProjectCreator, selection_arguments: tuple[str, ...], expect_envs: tuple[str, ...]) -> None:
    if False:
        return 10
    ini = '\n        [tox]\n        env_list = py3{10,9}-{django20,django21}{-cov,}\n        '
    project = tox_project({'tox.ini': ini})
    outcome = project.run('l', '--no-desc', *selection_arguments)
    outcome.assert_success()
    outcome.assert_out_err('{}\n'.format('\n'.join(expect_envs)), '')

def test_tox_skip_env(tox_project: ToxProjectCreator, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    monkeypatch.setenv('TOX_SKIP_ENV', 'm[y]py')
    project = tox_project({'tox.ini': '[tox]\nenv_list = py3{10,9},mypy'})
    outcome = project.run('l', '--no-desc', '-q')
    outcome.assert_success()
    outcome.assert_out_err('py310\npy39\n', '')

def test_tox_skip_env_cli(tox_project: ToxProjectCreator, monkeypatch: MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    monkeypatch.delenv('TOX_SKIP_ENV', raising=False)
    project = tox_project({'tox.ini': '[tox]\nenv_list = py3{10,9},mypy'})
    outcome = project.run('l', '--no-desc', '-q', '--skip-env', 'm[y]py')
    outcome.assert_success()
    outcome.assert_out_err('py310\npy39\n', '')

def test_tox_skip_env_logs(tox_project: ToxProjectCreator, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setenv('TOX_SKIP_ENV', 'm[y]py')
    project = tox_project({'tox.ini': '[tox]\nenv_list = py3{10,9},mypy'})
    outcome = project.run('l', '--no-desc')
    outcome.assert_success()
    outcome.assert_out_err("ROOT: skip environment mypy, matches filter 'm[y]py'\npy310\npy39\n", '')

def test_env_select_lazily_looks_at_envs() -> None:
    if False:
        while True:
            i = 10
    state = State(get_options(), [])
    env_selector = EnvSelector(state)
    state.conf.options.env = CliEnv('py')
    assert set(env_selector.iter()) == {'py'}

def test_cli_env_can_be_specified_in_default(tox_project: ToxProjectCreator) -> None:
    if False:
        for i in range(10):
            print('nop')
    proj = tox_project({'tox.ini': '[tox]\nenv_list=exists'})
    outcome = proj.run('r', '-e', 'exists')
    outcome.assert_success()
    assert 'exists' in outcome.out
    assert not outcome.err

def test_cli_env_can_be_specified_in_additional_environments(tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    proj = tox_project({'tox.ini': '[testenv:exists]'})
    outcome = proj.run('r', '-e', 'exists')
    outcome.assert_success()
    assert 'exists' in outcome.out
    assert not outcome.err

@pytest.mark.parametrize('env_name', ['py', CURRENT_PY_ENV, '.pkg'])
def test_allowed_implicit_cli_envs(env_name: str, tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    proj = tox_project({'tox.ini': ''})
    outcome = proj.run('r', '-e', env_name)
    outcome.assert_success()
    assert env_name in outcome.out
    assert not outcome.err

@pytest.mark.parametrize('env_name', ['a', 'b', 'a-b', 'b-a'])
def test_matches_hyphenated_env(env_name: str, tox_project: ToxProjectCreator) -> None:
    if False:
        i = 10
        return i + 15
    tox_ini = '\n        [tox]\n        env_list=a-b\n        [testenv]\n        package=skip\n        commands_pre =\n            a: python -c \'print("a")\'\n            b: python -c \'print("b")\'\n        commands=python -c \'print("ok")\'\n    '
    proj = tox_project({'tox.ini': tox_ini})
    outcome = proj.run('r', '-e', env_name)
    outcome.assert_success()
    assert env_name in outcome.out
    assert not outcome.err
_MINOR = sys.version_info.minor

@pytest.mark.parametrize('env_name', [f'3.{_MINOR}', f'3.{_MINOR}-cov', '3-cov', '3', f'py3.{_MINOR}', f'py3{_MINOR}-cov', f'py3.{_MINOR}-cov'])
def test_matches_combined_env(env_name: str, tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    tox_ini = '\n        [testenv]\n        package=skip\n        commands =\n            !cov: python -c \'print("without cov")\'\n            cov: python -c \'print("with cov")\'\n    '
    proj = tox_project({'tox.ini': tox_ini})
    outcome = proj.run('r', '-e', env_name)
    outcome.assert_success()
    assert env_name in outcome.out
    assert not outcome.err

@pytest.mark.parametrize('env', ['py', 'pypy', 'pypy3', 'pypy3.12', 'pypy312', 'py3', 'py3.12', 'py312', '3', '3.12', '3.12.0'])
def test_dynamic_env_factors_match(env: str) -> None:
    if False:
        return 10
    assert _DYNAMIC_ENV_FACTORS.fullmatch(env)

@pytest.mark.parametrize('env', ['cy3', 'cov', 'py10.1'])
def test_dynamic_env_factors_not_match(env: str) -> None:
    if False:
        i = 10
        return i + 15
    assert not _DYNAMIC_ENV_FACTORS.fullmatch(env)

def test_suggest_env(tox_project: ToxProjectCreator) -> None:
    if False:
        while True:
            i = 10
    tox_ini = f'[testenv:release]\n[testenv:py3{_MINOR}]\n[testenv:alpha-py3{_MINOR}]\n'
    proj = tox_project({'tox.ini': tox_ini})
    outcome = proj.run('r', '-e', f'releas,p3{_MINOR},magic,alph-p{_MINOR}')
    outcome.assert_failed(code=-2)
    assert not outcome.err
    msg = f'ROOT: HandledError| provided environments not found in configuration file:\nreleas - did you mean release?\np3{_MINOR} - did you mean py3{_MINOR}?\nmagic\nalph-p{_MINOR} - did you mean alpha-py3{_MINOR}?\n'
    assert outcome.out == msg