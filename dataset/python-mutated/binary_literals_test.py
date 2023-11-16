from __future__ import annotations
import pytest
from pyupgrade._main import _fix_tokens

@pytest.mark.parametrize('s', ('"☃".encode("UTF-8")', '"\\u2603".encode("UTF-8")', '"\\U0001f643".encode("UTF-8")', '"\\N{SNOWMAN}".encode("UTF-8")', '"\\xa0".encode("UTF-8")', '"y".encode("utf16")', 'f"{x}".encode()', '"foo".encode', '("foo".encode)', 'x.encode()', 'str.encode(f"{c}")', '"foo".encode(f"{c}")', pytest.param('wat.encode(b"unrelated")', id='unrelated .encode(...)')))
def test_binary_literals_noop(s):
    if False:
        return 10
    assert _fix_tokens(s) == s

@pytest.mark.parametrize(('s', 'expected'), (('"foo".encode()', 'b"foo"'), ('"foo".encode("ascii")', 'b"foo"'), ('"foo".encode("utf-8")', 'b"foo"'), ('"\\xa0".encode("latin1")', 'b"\\xa0"'), ('"\\\\u wot".encode()', 'b"\\\\u wot"'), ('"\\\\x files".encode()', 'b"\\\\x files"'), ('f(\n    "foo"\n    "bar".encode()\n)\n', 'f(\n    b"foo"\n    b"bar"\n)\n')))
def test_binary_literals(s, expected):
    if False:
        while True:
            i = 10
    assert _fix_tokens(s) == expected