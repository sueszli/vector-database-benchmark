from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', (pytest.param('try: ...\nexcept Exception:\n    raise', id='empty raise'), pytest.param('try: ...\nexcept: ...\n', id='empty try-except'), pytest.param('try: ...\nexcept AssertionError: ...\n', id='unrelated exception type as name'), pytest.param('try: ...\nexcept (AssertionError,): ...\n', id='unrelated exception type as tuple'), pytest.param('try: ...\nexcept OSError: ...\n', id='already rewritten name'), pytest.param('try: ...\nexcept (TypeError, OSError): ...\n', id='already rewritten tuple'), pytest.param('from .os import error\nraise error(1)\n', id='same name as rewrite but relative import'), pytest.param('from os import error\ndef f():\n    error = 3\n    return error\n', id='not rewriting outside of raise or except'), pytest.param('from os import error as the_roof\nraise the_roof()\n', id='ignoring imports with aliases'), pytest.param('import os\ntry: ...\nexcept (os).error: ...\n', id='weird parens')))
def test_fix_exceptions_noop(s):
    if False:
        return 10
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'version'), (pytest.param('raise socket.timeout()', (3, 9), id='raise socket.timeout is noop <3.10'), pytest.param('try: ...\nexcept socket.timeout: ...\n', (3, 9), id='except socket.timeout is noop <3.10'), pytest.param('raise asyncio.TimeoutError()', (3, 10), id='raise asyncio.TimeoutError() is noop <3.11'), pytest.param('try: ...\nexcept asyncio.TimeoutError: ...\n', (3, 10), id='except asyncio.TimeoutError() is noop <3.11')))
def test_fix_exceptions_version_specific_noop(s, version):
    if False:
        print('Hello World!')
    assert _fix_plugins(s, settings=Settings(min_version=version)) == s

@pytest.mark.parametrize(('s', 'expected'), (pytest.param('raise mmap.error(1)\n', 'raise OSError(1)\n', id='mmap.error'), pytest.param('raise os.error(1)\n', 'raise OSError(1)\n', id='os.error'), pytest.param('raise select.error(1)\n', 'raise OSError(1)\n', id='select.error'), pytest.param('raise socket.error(1)\n', 'raise OSError(1)\n', id='socket.error'), pytest.param('raise IOError(1)\n', 'raise OSError(1)\n', id='IOError'), pytest.param('raise EnvironmentError(1)\n', 'raise OSError(1)\n', id='EnvironmentError'), pytest.param('raise WindowsError(1)\n', 'raise OSError(1)\n', id='WindowsError'), pytest.param('raise os.error\n', 'raise OSError\n', id='raise exception type without call'), pytest.param('from os import error\nraise error(1)\n', 'from os import error\nraise OSError(1)\n', id='raise via from import'), pytest.param('try: ...\nexcept WindowsError: ...\n', 'try: ...\nexcept OSError: ...\n', id='except of name'), pytest.param('try: ...\nexcept os.error: ...\n', 'try: ...\nexcept OSError: ...\n', id='except of dotted name'), pytest.param('try: ...\nexcept (WindowsError,): ...\n', 'try: ...\nexcept OSError: ...\n', id='except of name in tuple'), pytest.param('try: ...\nexcept (os.error,): ...\n', 'try: ...\nexcept OSError: ...\n', id='except of dotted name in tuple'), pytest.param('try: ...\nexcept (WindowsError, KeyError, OSError): ...\n', 'try: ...\nexcept (OSError, KeyError): ...\n', id='deduplicates exception types'), pytest.param('try: ...\nexcept (os.error, WindowsError, OSError): ...\n', 'try: ...\nexcept OSError: ...\n', id='deduplicates to a single type'), pytest.param('try: ...\nexcept(os.error, WindowsError, OSError): ...\n', 'try: ...\nexcept OSError: ...\n', id='deduplicates to a single type without whitespace'), pytest.param('from wat import error\ntry: ...\nexcept (WindowsError, error): ...\n', 'from wat import error\ntry: ...\nexcept (OSError, error): ...\n', id='leave unrelated error names alone')))
def test_fix_exceptions(s, expected):
    if False:
        print('Hello World!')
    assert _fix_plugins(s, settings=Settings()) == expected

@pytest.mark.parametrize(('s', 'expected', 'version'), (pytest.param('raise socket.timeout(1)\n', 'raise TimeoutError(1)\n', (3, 10), id='socket.timeout'), pytest.param('raise asyncio.TimeoutError(1)\n', 'raise TimeoutError(1)\n', (3, 11), id='asyncio.TimeoutError')))
def test_fix_exceptions_versioned(s, expected, version):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings(min_version=version)) == expected

def test_can_rewrite_disparate_names():
    if False:
        while True:
            i = 10
    s = 'try: ...\nexcept (asyncio.TimeoutError, WindowsError): ...\n'
    expected = 'try: ...\nexcept (TimeoutError, OSError): ...\n'
    assert _fix_plugins(s, settings=Settings(min_version=(3, 11))) == expected