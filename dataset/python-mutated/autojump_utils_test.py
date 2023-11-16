import os
import sys
import mock
import pytest
sys.path.append(os.path.join(os.getcwd(), 'bin'))
import autojump_utils
from autojump_utils import encode_local
from autojump_utils import first
from autojump_utils import get_tab_entry_info
from autojump_utils import has_uppercase
from autojump_utils import in_bash
from autojump_utils import is_python3
from autojump_utils import last
from autojump_utils import sanitize
from autojump_utils import second
from autojump_utils import surround_quotes
from autojump_utils import take
from autojump_utils import unico
if is_python3():
    os.getcwdu = os.getcwd
    xrange = range

def u(string):
    if False:
        i = 10
        return i + 15
    "\n    This is a unicode() wrapper since u'string' is a Python3 compiler error.\n    "
    if is_python3():
        return string
    return unicode(string, encoding='utf-8', errors='strict')

@pytest.mark.skipif(is_python3(), reason='Unicode sucks.')
@mock.patch.object(sys, 'getfilesystemencoding', return_value='ascii')
def test_encode_local_ascii(_):
    if False:
        i = 10
        return i + 15
    assert encode_local(u('foo')) == b'foo'

@pytest.mark.skipif(is_python3(), reason='Unicode sucks.')
@pytest.mark.xfail(reason='disabled due to pytest bug: https://bitbucket.org/hpk42/pytest/issue/534/pytest-fails-to-catch-unicodedecodeerrors')
@mock.patch.object(sys, 'getfilesystemencoding', return_value='ascii')
def test_encode_local_ascii_fails(_):
    if False:
        return 10
    with pytest.raises(UnicodeDecodeError):
        encode_local(u('日本語'))

@pytest.mark.skipif(is_python3(), reason='Unicode sucks.')
@mock.patch.object(sys, 'getfilesystemencoding', return_value=None)
def test_encode_local_empty(_):
    if False:
        for i in range(10):
            print('nop')
    assert encode_local(b'foo') == u('foo')

@pytest.mark.skipif(is_python3(), reason='Unicode sucks.')
@mock.patch.object(sys, 'getfilesystemencoding', return_value='utf-8')
def test_encode_local_unicode(_):
    if False:
        for i in range(10):
            print('nop')
    assert encode_local(b'foo') == u('foo')
    assert encode_local(u('foo')) == u('foo')

def test_has_uppercase():
    if False:
        i = 10
        return i + 15
    assert has_uppercase('Foo')
    assert has_uppercase('foO')
    assert not has_uppercase('foo')
    assert not has_uppercase('')

@mock.patch.object(autojump_utils, 'in_bash', return_value=True)
def test_surround_quotes_in_bash(_):
    if False:
        for i in range(10):
            print('nop')
    assert surround_quotes('foo') == '"foo"'

@mock.patch.object(autojump_utils, 'in_bash', return_value=False)
def test_dont_surround_quotes_not_in_bash(_):
    if False:
        return 10
    assert surround_quotes('foo') == 'foo'

def test_sanitize():
    if False:
        return 10
    assert sanitize([]) == []
    assert sanitize(['/foo/bar/', '/']) == [u('/foo/bar'), u('/')]

@pytest.mark.skipif(is_python3(), reason='Unicode sucks.')
def test_unico():
    if False:
        for i in range(10):
            print('nop')
    assert unico(str('blah')) == u('blah')
    assert unico(str('日本語')) == u('日本語')
    assert unico(u('でもおれは中国人だ。')) == u('でもおれは中国人だ。')

def test_first():
    if False:
        print('Hello World!')
    assert first(xrange(5)) == 0
    assert first([]) is None

def test_second():
    if False:
        print('Hello World!')
    assert second(xrange(5)) == 1
    assert second([]) is None

def test_last():
    if False:
        i = 10
        return i + 15
    assert last(xrange(4)) == 3
    assert last([]) is None

def test_take():
    if False:
        return 10
    assert list(take(1, xrange(3))) == [0]
    assert list(take(2, xrange(3))) == [0, 1]
    assert list(take(4, xrange(3))) == [0, 1, 2]
    assert list(take(10, [])) == []

def test_in_bash():
    if False:
        while True:
            i = 10
    for path in ['/bin/bash', '/usr/bin/bash']:
        os.environ['SHELL'] = path
        assert in_bash()
    for path in ['/bin/zsh', '/usr/bin/zsh']:
        os.environ['SHELL'] = '/usr/bin/zsh'
        assert not in_bash()

def test_get_needle():
    if False:
        return 10
    assert get_tab_entry_info('foo__', '__') == ('foo', None, None)

def test_get_index():
    if False:
        i = 10
        return i + 15
    assert get_tab_entry_info('foo__2', '__') == ('foo', 2, None)

def test_get_path():
    if False:
        while True:
            i = 10
    assert get_tab_entry_info('foo__3__/foo/bar', '__') == ('foo', 3, '/foo/bar')

def test_get_none():
    if False:
        i = 10
        return i + 15
    assert get_tab_entry_info('gibberish content', '__') == (None, None, None)