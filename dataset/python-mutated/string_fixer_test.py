from __future__ import annotations
import textwrap
import pytest
from pre_commit_hooks.string_fixer import main
TESTS = (("''", "''", 0), ('""', "''", 1), ('"\\\'"', '"\\\'"', 0), ('"\\""', '"\\""', 0), ('\'\\"\\"\'', '\'\\"\\"\'', 0), ('x = "foo"', "x = 'foo'", 1), ('"\\\'"', '"\\\'"', 0), ('""" Foo """', '""" Foo """', 0), (textwrap.dedent('\n        x = " \\\n        foo \\\n        "\n\n        '), textwrap.dedent("\n        x = ' \\\n        foo \\\n        '\n\n        "), 1), ('"foo""bar"', "'foo''bar'", 1), pytest.param('f\'hello{"world"}\'', 'f\'hello{"world"}\'', 0, id='ignore nested fstrings'))

@pytest.mark.parametrize(('input_s', 'output', 'expected_retval'), TESTS)
def test_rewrite(input_s, output, expected_retval, tmpdir):
    if False:
        i = 10
        return i + 15
    path = tmpdir.join('file.py')
    path.write(input_s)
    retval = main([str(path)])
    assert path.read() == output
    assert retval == expected_retval

def test_rewrite_crlf(tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('f.py')
    f.write_binary(b'"foo"\r\n"bar"\r\n')
    assert main((str(f),))
    assert f.read_binary() == b"'foo'\r\n'bar'\r\n"