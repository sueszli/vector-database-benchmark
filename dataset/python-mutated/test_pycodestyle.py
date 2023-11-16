import pytest
from pycodestyle import expand_indent
from pycodestyle import mute_string

@pytest.mark.parametrize(('s', 'expected'), (('    ', 4), ('\t', 8), ('       \t', 8), ('        \t', 16)))
def test_expand_indent(s, expected):
    if False:
        i = 10
        return i + 15
    assert expand_indent(s) == expected

@pytest.mark.parametrize(('s', 'expected'), (('"abc"', '"xxx"'), ("'''abc'''", "'''xxx'''"), ("r'abc'", "r'xxx'")))
def test_mute_string(s, expected):
    if False:
        i = 10
        return i + 15
    assert mute_string(s) == expected