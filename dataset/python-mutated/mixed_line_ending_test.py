from __future__ import annotations
import pytest
from pre_commit_hooks.mixed_line_ending import main

@pytest.mark.parametrize(('input_s', 'output'), ((b'foo\r\nbar\nbaz\n', b'foo\nbar\nbaz\n'), (b'foo\r\nbar\nbaz\r\n', b'foo\r\nbar\r\nbaz\r\n'), (b'foo\rbar\nbaz\r', b'foo\rbar\rbaz\r'), (b'foo\r\nbar\n', b'foo\nbar\n'), (b'foo\rbar\n', b'foo\nbar\n'), (b'foo\r\nbar\r', b'foo\r\nbar\r\n'), (b'foo\r\nbar\nbaz\r', b'foo\nbar\nbaz\n')))
def test_mixed_line_ending_fixes_auto(input_s, output, tmpdir):
    if False:
        print('Hello World!')
    path = tmpdir.join('file.txt')
    path.write_binary(input_s)
    ret = main((str(path),))
    assert ret == 1
    assert path.read_binary() == output

def test_non_mixed_no_newline_end_of_file(tmpdir):
    if False:
        print('Hello World!')
    path = tmpdir.join('f.txt')
    path.write_binary(b'foo\nbar\nbaz')
    assert not main((str(path),))
    assert path.read_binary() == b'foo\nbar\nbaz'

def test_mixed_no_newline_end_of_file(tmpdir):
    if False:
        print('Hello World!')
    path = tmpdir.join('f.txt')
    path.write_binary(b'foo\r\nbar\nbaz')
    assert main((str(path),))
    assert path.read_binary() == b'foo\nbar\nbaz\n'

@pytest.mark.parametrize(('fix_option', 'input_s'), (('--fix=auto', b'foo\r\nbar\r\nbaz\r\n'), ('--fix=auto', b'foo\rbar\rbaz\r'), ('--fix=auto', b'foo\nbar\nbaz\n'), ('--fix=crlf', b'foo\r\nbar\r\nbaz\r\n'), ('--fix=lf', b'foo\nbar\nbaz\n')))
def test_line_endings_ok(fix_option, input_s, tmpdir, capsys):
    if False:
        print('Hello World!')
    path = tmpdir.join('input.txt')
    path.write_binary(input_s)
    ret = main((fix_option, str(path)))
    assert ret == 0
    assert path.read_binary() == input_s
    (out, _) = capsys.readouterr()
    assert out == ''

def test_no_fix_does_not_modify(tmpdir, capsys):
    if False:
        i = 10
        return i + 15
    path = tmpdir.join('input.txt')
    contents = b'foo\r\nbar\rbaz\nwomp\n'
    path.write_binary(contents)
    ret = main(('--fix=no', str(path)))
    assert ret == 1
    assert path.read_binary() == contents
    (out, _) = capsys.readouterr()
    assert out == f'{path}: mixed line endings\n'

def test_fix_lf(tmpdir, capsys):
    if False:
        for i in range(10):
            print('nop')
    path = tmpdir.join('input.txt')
    path.write_binary(b'foo\r\nbar\rbaz\n')
    ret = main(('--fix=lf', str(path)))
    assert ret == 1
    assert path.read_binary() == b'foo\nbar\nbaz\n'
    (out, _) = capsys.readouterr()
    assert out == f'{path}: fixed mixed line endings\n'

def test_fix_crlf(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    path = tmpdir.join('input.txt')
    path.write_binary(b'foo\r\nbar\rbaz\n')
    ret = main(('--fix=crlf', str(path)))
    assert ret == 1
    assert path.read_binary() == b'foo\r\nbar\r\nbaz\r\n'

def test_fix_lf_all_crlf(tmpdir):
    if False:
        while True:
            i = 10
    'Regression test for #239'
    path = tmpdir.join('input.txt')
    path.write_binary(b'foo\r\nbar\r\n')
    ret = main(('--fix=lf', str(path)))
    assert ret == 1
    assert path.read_binary() == b'foo\nbar\n'