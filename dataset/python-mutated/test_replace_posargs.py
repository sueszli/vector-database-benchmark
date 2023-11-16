from __future__ import annotations
import sys
from typing import TYPE_CHECKING
import pytest
if TYPE_CHECKING:
    from tests.config.loader.ini.replace.conftest import ReplaceOne

@pytest.mark.parametrize('syntax', ['{posargs}', '[]'])
def test_replace_pos_args_none_sys_argv(syntax: str, replace_one: ReplaceOne) -> None:
    if False:
        return 10
    result = replace_one(syntax, None)
    assert not result

@pytest.mark.parametrize('syntax', ['{posargs}', '[]'])
def test_replace_pos_args_empty_sys_argv(syntax: str, replace_one: ReplaceOne) -> None:
    if False:
        while True:
            i = 10
    result = replace_one(syntax, [])
    assert not result

@pytest.mark.parametrize('syntax', ['{posargs}', '[]'])
def test_replace_pos_args_extra_sys_argv(syntax: str, replace_one: ReplaceOne) -> None:
    if False:
        while True:
            i = 10
    result = replace_one(syntax, [sys.executable, 'magic'])
    assert result == f'{sys.executable} magic'

@pytest.mark.parametrize('syntax', ['{posargs}', '[]'])
def test_replace_pos_args(syntax: str, replace_one: ReplaceOne) -> None:
    if False:
        for i in range(10):
            print('nop')
    result = replace_one(syntax, ['ok', 'what', ' yes '])
    quote = '"' if sys.platform == 'win32' else "'"
    assert result == f'ok what {quote} yes {quote}'

@pytest.mark.parametrize(('value', 'result'), [('magic', 'magic'), ('magic:colon', 'magic:colon'), ('magic\n b c', 'magic\nb c'), ('magi\\\n c d', 'magic d'), ('\\{a\\}', '{a}')])
def test_replace_pos_args_default(replace_one: ReplaceOne, value: str, result: str) -> None:
    if False:
        return 10
    outcome = replace_one(f'{{posargs:{value}}}', None)
    assert result == outcome

@pytest.mark.parametrize('value', ['\\{posargs}', '{posargs\\}', '\\{posargs\\}', '{\\{posargs}', '{\\{posargs}{}', '\\[]', '[\\]', '\\[\\]'])
def test_replace_pos_args_escaped(replace_one: ReplaceOne, value: str) -> None:
    if False:
        i = 10
        return i + 15
    result = replace_one(value, None)
    outcome = value.replace('\\', '')
    assert result == outcome

@pytest.mark.parametrize(('value', 'result'), [('[]-{posargs}', 'foo-foo'), ('{posargs}-[]', 'foo-foo')])
def test_replace_mixed_brackets_and_braces(replace_one: ReplaceOne, value: str, result: str) -> None:
    if False:
        print('Hello World!')
    outcome = replace_one(value, ['foo'])
    assert result == outcome

def test_half_escaped_braces(replace_one: ReplaceOne) -> None:
    if False:
        return 10
    'See https://github.com/tox-dev/tox/issues/1956'
    outcome = replace_one('\\{posargs} {posargs}', ['foo'])
    assert outcome == '{posargs} foo'