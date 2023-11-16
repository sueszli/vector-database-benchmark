from __future__ import annotations
import pytest
from pyupgrade._main import _fix_tokens

@pytest.mark.parametrize('s', (pytest.param('(', id='syntax errors are unchanged'), pytest.param('"""with newline\n"""', id='string containing newline'), pytest.param('def f():\n    return"foo"\n', id='Regression: no space between return and string')))
def test_unicode_literals_noop(s):
    if False:
        return 10
    assert _fix_tokens(s) == s

@pytest.mark.parametrize(('s', 'expected'), (pytest.param("u''", "''", id='it removes u prefix'),))
def test_unicode_literals(s, expected):
    if False:
        for i in range(10):
            print('nop')
    assert _fix_tokens(s) == expected