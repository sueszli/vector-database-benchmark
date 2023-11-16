from __future__ import annotations
import pytest
from pyupgrade._main import _fix_tokens

@pytest.mark.parametrize('s', ('""', 'r"\\d"', "r'\\d'", 'r"""\\d"""', "r'''\\d'''", 'rb"\\d"', '"\\\\d"', '"\\u2603"', '"\\r\\n"', '"\\N{SNOWMAN}"', '"""\\\n"""', '"""\\\r\n"""', '"""\\\r"""'))
def test_fix_escape_sequences_noop(s):
    if False:
        while True:
            i = 10
    assert _fix_tokens(s) == s

@pytest.mark.parametrize(('s', 'expected'), (('"\\d"', 'r"\\d"'), ('"\\n\\d"', '"\\n\\\\d"'), ('u"\\d"', 'r"\\d"'), ('b"\\d"', 'br"\\d"'), ('"\\8"', 'r"\\8"'), ('"\\9"', 'r"\\9"'), ('b"\\u2603"', 'br"\\u2603"'), ('"""\\\n\\q"""', '"""\\\n\\\\q"""'), ('"""\\\r\n\\q"""', '"""\\\r\n\\\\q"""'), ('"""\\\r\\q"""', '"""\\\r\\\\q"""'), ('"\\N"', 'r"\\N"'), ('"\\N\\n"', '"\\\\N\\n"'), ('"\\N{SNOWMAN}\\q"', '"\\N{SNOWMAN}\\\\q"'), ('b"\\N{SNOWMAN}"', 'br"\\N{SNOWMAN}"')))
def test_fix_escape_sequences(s, expected):
    if False:
        return 10
    assert _fix_tokens(s) == expected