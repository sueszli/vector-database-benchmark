from __future__ import annotations
import pytest
from pre_commit_hooks.check_docstring_first import check_docstring_first
from pre_commit_hooks.check_docstring_first import main
TESTS = ((b'', 0, ''), (b'"foo"', 0, ''), (b'from __future__ import unicode_literals\n"foo"\n', 1, '{filename}:2: Module docstring appears after code (code seen on line 1).\n'), (b'"The real docstring"\nfrom __future__ import absolute_import\n"fake docstring"\n', 1, '{filename}:3: Multiple module docstrings (first docstring on line 1).\n'), (b'import os\nimport sys\n"docstring"\n', 1, '{filename}:3: Module docstring appears after code (code seen on line 1).\n'), (b'x = "foo"\n', 0, ''))
all_tests = pytest.mark.parametrize(('contents', 'expected', 'expected_out'), TESTS)

@all_tests
def test_unit(capsys, contents, expected, expected_out):
    if False:
        for i in range(10):
            print('nop')
    assert check_docstring_first(contents) == expected
    assert capsys.readouterr()[0] == expected_out.format(filename='<unknown>')

@all_tests
def test_integration(tmpdir, capsys, contents, expected, expected_out):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('test.py')
    f.write_binary(contents)
    assert main([str(f)]) == expected
    assert capsys.readouterr()[0] == expected_out.format(filename=str(f))

def test_arbitrary_encoding(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('f.py')
    contents = '# -*- coding: cp1252\nx = "£"'.encode('cp1252')
    f.write_binary(contents)
    assert main([str(f)]) == 0