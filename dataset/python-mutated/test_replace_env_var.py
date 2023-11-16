from __future__ import annotations
import threading
from typing import TYPE_CHECKING, Generator
import pytest
if TYPE_CHECKING:
    from tests.config.loader.ini.replace.conftest import ReplaceOne
    from tox.pytest import LogCaptureFixture, MonkeyPatch

def test_replace_env_set(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    "If we have a factor that is not specified within the core env-list then that's also an environment"
    monkeypatch.setenv('MAGIC', 'something good')
    result = replace_one('{env:MAGIC}')
    assert result == 'something good'

def test_replace_env_set_double_bs(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    'Double backslash should remain but not affect surrounding replacements.'
    monkeypatch.setenv('MAGIC', 'something good')
    result = replace_one('{env:MAGIC}\\\\{env:MAGIC}')
    assert result == 'something good\\\\something good'

def test_replace_env_set_triple_bs(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    'Triple backslash should retain two slashes with the third escaping subsequent replacement.'
    monkeypatch.setenv('MAGIC', 'something good')
    result = replace_one('{env:MAGIC}\\\\\\{env:MAGIC}')
    assert result == 'something good\\\\{env:MAGIC}'

def test_replace_env_set_quad_bs(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    'Quad backslash should remain but not affect surrounding replacements.'
    monkeypatch.setenv('MAGIC', 'something good')
    result = replace_one('\\\\{env:MAGIC}\\\\\\\\{env:MAGIC}' + '\\')
    assert result == '\\\\something good\\\\\\\\something good' + '\\'

def test_replace_env_when_value_is_backslash(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    "When the replacement value is backslash, it shouldn't affect the next replacement."
    monkeypatch.setenv('MAGIC', 'tragic')
    monkeypatch.setenv('BS', '\\')
    result = replace_one('{env:BS}{env:MAGIC}')
    assert result == '\\tragic'

def test_replace_env_when_value_is_stuff_then_backslash(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        i = 10
        return i + 15
    "When the replacement value is a string containing backslash, it shouldn't affect the next replacement."
    monkeypatch.setenv('MAGIC', 'tragic')
    monkeypatch.setenv('BS', 'stuff\\')
    result = replace_one('{env:BS}{env:MAGIC}')
    assert result == 'stuff\\tragic'

def test_replace_env_missing(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    "If we have a factor that is not specified within the core env-list then that's also an environment"
    monkeypatch.delenv('MAGIC', raising=False)
    result = replace_one('{env:MAGIC}')
    assert not result

def test_replace_env_missing_default(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    "If we have a factor that is not specified within the core env-list then that's also an environment"
    monkeypatch.delenv('MAGIC', raising=False)
    result = replace_one('{env:MAGIC:def}')
    assert result == 'def'

def test_replace_env_missing_default_from_env(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    "If we have a factor that is not specified within the core env-list then that's also an environment"
    monkeypatch.delenv('MAGIC', raising=False)
    monkeypatch.setenv('MAGIC_DEFAULT', 'yes')
    result = replace_one('{env:MAGIC:{env:MAGIC_DEFAULT}}')
    assert result == 'yes'

def test_replace_env_var_multiple(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    'Multiple env substitutions on a single line.'
    monkeypatch.setenv('MAGIC', 'MAGIC')
    monkeypatch.setenv('TRAGIC', 'TRAGIC')
    result = replace_one('{env:MAGIC} {env:TRAGIC} {env:MAGIC}')
    assert result == 'MAGIC TRAGIC MAGIC'

def test_replace_env_var_multiple_default(replace_one: ReplaceOne) -> None:
    if False:
        print('Hello World!')
    'Multiple env substitutions on a single line with default values.'
    result = replace_one('{env:MAGIC:foo} {env:TRAGIC:bar} {env:MAGIC:baz}')
    assert result == 'foo bar baz'

def test_replace_env_var_circular(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    'Replacement values will not infinitely loop'
    monkeypatch.setenv('MAGIC', '{env:MAGIC}')
    result = replace_one('{env:MAGIC}')
    assert result == '{env:MAGIC}'

@pytest.fixture()
def reset_env_var_after_delay(monkeypatch: MonkeyPatch) -> Generator[threading.Thread, None, None]:
    if False:
        return 10
    timeout = 2

    def avoid_infinite_loop() -> None:
        if False:
            return 10
        monkeypatch.setenv('TRAGIC', f'envvar forcibly reset after {timeout} sec')
    timer = threading.Timer(2, avoid_infinite_loop)
    timer.start()
    yield timer
    timer.cancel()
    timer.join()

@pytest.mark.usefixtures('reset_env_var_after_delay')
def test_replace_env_var_circular_flip_flop(replace_one: ReplaceOne, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
    if False:
        print('Hello World!')
    'Replacement values will not infinitely loop back and forth'
    monkeypatch.setenv('TRAGIC', '{env:MAGIC}')
    monkeypatch.setenv('MAGIC', '{env:TRAGIC}')
    result = replace_one('{env:MAGIC}')
    assert result == '{env:MAGIC}'
    assert 'circular chain between set env MAGIC, TRAGIC' in caplog.messages

@pytest.mark.usefixtures('reset_env_var_after_delay')
def test_replace_env_var_circular_flip_flop_5(replace_one: ReplaceOne, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    'Replacement values will not infinitely loop back and forth (longer chain)'
    monkeypatch.setenv('MAGIC', '{env:TRAGIC}')
    monkeypatch.setenv('TRAGIC', '{env:RABBIT}')
    monkeypatch.setenv('RABBIT', '{env:HAT}')
    monkeypatch.setenv('HAT', '{env:TRICK}')
    monkeypatch.setenv('TRICK', '{env:MAGIC}')
    result = replace_one('{env:MAGIC}')
    assert result == '{env:MAGIC}'
    assert 'circular chain between set env MAGIC, TRAGIC, RABBIT, HAT, TRICK' in caplog.messages

@pytest.mark.parametrize('fallback', [True, False])
def test_replace_env_var_chase(replace_one: ReplaceOne, monkeypatch: MonkeyPatch, fallback: bool) -> None:
    if False:
        return 10
    'Resolve variable to be replaced and default value via indirection.'
    monkeypatch.setenv('WALK', 'THIS')
    def_val = 'or that one'
    monkeypatch.setenv('DEF', def_val)
    if fallback:
        monkeypatch.delenv('THIS', raising=False)
        exp_result = def_val
    else:
        this_val = 'path'
        monkeypatch.setenv('THIS', this_val)
        exp_result = this_val
    result = replace_one('{env:{env:WALK}:{env:DEF}}')
    assert result == exp_result

def test_replace_env_default_with_colon(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    "If we have a factor that is not specified within the core env-list then that's also an environment"
    monkeypatch.delenv('MAGIC', raising=False)
    result = replace_one('{env:MAGIC:https://some.url.org}')
    assert result == 'https://some.url.org'

def test_replace_env_default_deep(replace_one: ReplaceOne, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Get the value through a long tree of nested defaults.'
    monkeypatch.delenv('M', raising=False)
    assert replace_one('{env:M:{env:M:{env:M:{env:M:{env:M:foo}}}}}') == 'foo'