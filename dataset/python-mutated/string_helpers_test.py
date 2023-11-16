from __future__ import annotations
import pytest
from pyupgrade._string_helpers import parse_format
from pyupgrade._string_helpers import unparse_parsed_string

@pytest.mark.parametrize('s', ('', 'foo', '{}', '{0}', '{named}', '{!r}', '{:>5}', '{{', '}}', '{0!s:15}'))
def test_roundtrip_text(s):
    if False:
        print('Hello World!')
    assert unparse_parsed_string(parse_format(s)) == s

def test_parse_format_starts_with_named():
    if False:
        return 10
    assert parse_format('\\N{snowman} hi {0} hello') == [('\\N{snowman} hi ', '0', '', None), (' hello', None, None, None)]

@pytest.mark.parametrize(('s', 'expected'), (('{:}', '{}'), ('{0:}', '{0}'), ('{0!r:}', '{0!r}')))
def test_intentionally_not_round_trip(s, expected):
    if False:
        while True:
            i = 10
    ret = unparse_parsed_string(parse_format(s))
    assert ret == expected