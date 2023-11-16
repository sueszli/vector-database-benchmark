from __future__ import annotations
from itertools import chain
import pytest
VALID_STRINGS = (("'a'", 'a'), ("'1'", '1'), ('1', 1), ('True', True), ('False', False), ('{}', {}))
NONSTRINGS = (({'a': 1}, {'a': 1}),)
INVALID_STRINGS = (('a=1', 'a=1', SyntaxError), ('a.foo()', 'a.foo()', None), ('import foo', 'import foo', None), ("__import__('foo')", "__import__('foo')", ValueError))

@pytest.mark.parametrize('code, expected, stdin', ((c, e, {}) for (c, e) in chain(VALID_STRINGS, NONSTRINGS)), indirect=['stdin'])
def test_simple_types(am, code, expected):
    if False:
        while True:
            i = 10
    assert am.safe_eval(code) == expected

@pytest.mark.parametrize('code, expected, stdin', ((c, e, {}) for (c, e) in chain(VALID_STRINGS, NONSTRINGS)), indirect=['stdin'])
def test_simple_types_with_exceptions(am, code, expected):
    if False:
        while True:
            i = 10
    assert am.safe_eval(code, include_exceptions=True), (expected, None)

@pytest.mark.parametrize('code, expected, stdin', ((c, e, {}) for (c, e, dummy) in INVALID_STRINGS), indirect=['stdin'])
def test_invalid_strings(am, code, expected):
    if False:
        return 10
    assert am.safe_eval(code) == expected

@pytest.mark.parametrize('code, expected, exception, stdin', ((c, e, ex, {}) for (c, e, ex) in INVALID_STRINGS), indirect=['stdin'])
def test_invalid_strings_with_exceptions(am, code, expected, exception):
    if False:
        print('Hello World!')
    res = am.safe_eval(code, include_exceptions=True)
    assert res[0] == expected
    if exception is None:
        assert res[1] == exception
    else:
        assert isinstance(res[1], exception)