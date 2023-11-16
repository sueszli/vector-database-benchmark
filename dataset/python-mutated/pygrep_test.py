from __future__ import annotations
import pytest
from pre_commit.languages import pygrep
from testing.language_helpers import run_language

@pytest.fixture
def some_files(tmpdir):
    if False:
        i = 10
        return i + 15
    tmpdir.join('f1').write_binary(b'foo\nbar\n')
    tmpdir.join('f2').write_binary(b'[INFO] hi\n')
    tmpdir.join('f3').write_binary(b"with'quotes\n")
    tmpdir.join('f4').write_binary(b'foo\npattern\nbar\n')
    tmpdir.join('f5').write_binary(b'[INFO] hi\npattern\nbar')
    tmpdir.join('f6').write_binary(b"pattern\nbarwith'foo\n")
    tmpdir.join('f7').write_binary(b"hello'hi\nworld\n")
    tmpdir.join('f8').write_binary(b'foo\nbar\nbaz\n')
    tmpdir.join('f9').write_binary(b'[WARN] hi\n')
    with tmpdir.as_cwd():
        yield

@pytest.mark.usefixtures('some_files')
@pytest.mark.parametrize(('pattern', 'expected_retcode', 'expected_out'), (('baz', 0, ''), ('foo', 1, 'f1:1:foo\n'), ('bar', 1, 'f1:2:bar\n'), ('(?i)\\[info\\]', 1, 'f2:1:[INFO] hi\n'), ("h'q", 1, "f3:1:with'quotes\n")))
def test_main(cap_out, pattern, expected_retcode, expected_out):
    if False:
        i = 10
        return i + 15
    ret = pygrep.main((pattern, 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == expected_retcode
    assert out == expected_out

@pytest.mark.usefixtures('some_files')
def test_negate_by_line_no_match(cap_out):
    if False:
        while True:
            i = 10
    ret = pygrep.main(('pattern\nbar', 'f4', 'f5', 'f6', '--negate'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f4\nf5\nf6\n'

@pytest.mark.usefixtures('some_files')
def test_negate_by_line_two_match(cap_out):
    if False:
        i = 10
        return i + 15
    ret = pygrep.main(('foo', 'f4', 'f5', 'f6', '--negate'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f5\n'

@pytest.mark.usefixtures('some_files')
def test_negate_by_line_all_match(cap_out):
    if False:
        while True:
            i = 10
    ret = pygrep.main(('pattern', 'f4', 'f5', 'f6', '--negate'))
    out = cap_out.get()
    assert ret == 0
    assert out == ''

@pytest.mark.usefixtures('some_files')
def test_negate_by_file_no_match(cap_out):
    if False:
        print('Hello World!')
    ret = pygrep.main(('baz', 'f4', 'f5', 'f6', '--negate', '--multiline'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f4\nf5\nf6\n'

@pytest.mark.usefixtures('some_files')
def test_negate_by_file_one_match(cap_out):
    if False:
        for i in range(10):
            print('nop')
    ret = pygrep.main(('foo\npattern', 'f4', 'f5', 'f6', '--negate', '--multiline'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f5\nf6\n'

@pytest.mark.usefixtures('some_files')
def test_negate_by_file_all_match(cap_out):
    if False:
        for i in range(10):
            print('nop')
    ret = pygrep.main(('pattern\nbar', 'f4', 'f5', 'f6', '--negate', '--multiline'))
    out = cap_out.get()
    assert ret == 0
    assert out == ''

@pytest.mark.usefixtures('some_files')
def test_ignore_case(cap_out):
    if False:
        return 10
    ret = pygrep.main(('--ignore-case', 'info', 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f2:1:[INFO] hi\n'

@pytest.mark.usefixtures('some_files')
def test_multiline(cap_out):
    if False:
        i = 10
        return i + 15
    ret = pygrep.main(('--multiline', 'foo\\nbar', 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f1:1:foo\nbar\n'

@pytest.mark.usefixtures('some_files')
def test_multiline_line_number(cap_out):
    if False:
        for i in range(10):
            print('nop')
    ret = pygrep.main(('--multiline', 'ar', 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f1:2:bar\n'

@pytest.mark.usefixtures('some_files')
def test_multiline_dotall_flag_is_enabled(cap_out):
    if False:
        return 10
    ret = pygrep.main(('--multiline', 'o.*bar', 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f1:1:foo\nbar\n'

@pytest.mark.usefixtures('some_files')
def test_multiline_multiline_flag_is_enabled(cap_out):
    if False:
        return 10
    ret = pygrep.main(('--multiline', 'foo$.*bar', 'f1', 'f2', 'f3'))
    out = cap_out.get()
    assert ret == 1
    assert out == 'f1:1:foo\nbar\n'

def test_grep_hook_matching(some_files, tmp_path):
    if False:
        for i in range(10):
            print('nop')
    ret = run_language(tmp_path, pygrep, 'ello', file_args=('f7', 'f8', 'f9'))
    assert ret == (1, b"f7:1:hello'hi\n")

@pytest.mark.parametrize('regex', ('nope', "foo'bar", '^\\[INFO\\]'))
def test_grep_hook_not_matching(regex, some_files, tmp_path):
    if False:
        return 10
    ret = run_language(tmp_path, pygrep, regex, file_args=('f7', 'f8', 'f9'))
    assert ret == (0, b'')