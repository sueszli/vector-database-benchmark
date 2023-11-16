"""Tests for qutebrowser.misc.crashdialog."""
import os
import pytest
from qutebrowser.misc import crashdialog
VALID_CRASH_TEXT = '\nFatal Python error: Segmentation fault\n_\nCurrent thread 0x00007f09b538d700 (most recent call first):\n  File "", line 1 in testfunc\n  File "filename", line 88 in func\n'
VALID_CRASH_TEXT_EMPTY = '\nFatal Python error: Aborted\n_\nCurrent thread 0x00007f09b538d700 (most recent call first):\n  File "", line 1 in_\n  File "filename", line 88 in func\n'
VALID_CRASH_TEXT_THREAD = '\nFatal Python error: Segmentation fault\n_\nThread 0x00007fa135ac7700 (most recent call first):\n  File "", line 1 in testfunc\n'
WINDOWS_CRASH_TEXT = '\nWindows fatal exception: access violation\n_\nCurrent thread 0x000014bc (most recent call first):\n  File "qutebrowser\\mainwindow\\tabbedbrowser.py", line 468 in tabopen\n  File "qutebrowser\\browser\\shared.py", line 247 in get_tab\n'
INVALID_CRASH_TEXT = '\nHello world!\n'

@pytest.mark.parametrize('text, typ, func', [(VALID_CRASH_TEXT, 'Segmentation fault', 'testfunc'), (VALID_CRASH_TEXT_THREAD, 'Segmentation fault', 'testfunc'), (VALID_CRASH_TEXT_EMPTY, 'Aborted', ''), (WINDOWS_CRASH_TEXT, 'Windows access violation', 'tabopen'), (INVALID_CRASH_TEXT, '', '')])
def test_parse_fatal_stacktrace(text, typ, func):
    if False:
        i = 10
        return i + 15
    text = text.strip().replace('_', ' ')
    assert crashdialog.parse_fatal_stacktrace(text) == (typ, func)

@pytest.mark.parametrize('env, expected', [({'FOO': 'bar'}, ''), ({'FOO': 'bar', 'LC_ALL': 'baz'}, 'LC_ALL = baz'), ({'LC_ALL': 'baz', 'PYTHONFOO': 'fish'}, 'LC_ALL = baz\nPYTHONFOO = fish'), ({'DE': 'KDE', 'DESKTOP_SESSION': 'plasma'}, 'DE = KDE\nDESKTOP_SESSION = plasma'), ({'QT5_IM_MODULE': 'fcitx', 'QT_IM_MODULE': 'fcitx'}, 'QT_IM_MODULE = fcitx'), ({'LANGUAGE': 'foo', 'LANG': 'en_US.UTF-8'}, 'LANG = en_US.UTF-8'), ({'FOO': 'bar', 'QUTE_BLAH': '1'}, 'QUTE_BLAH = 1')])
def test_get_environment_vars(monkeypatch, env, expected):
    if False:
        for i in range(10):
            print('nop')
    'Test for crashdialog._get_environment_vars.'
    for key in os.environ.copy():
        monkeypatch.delenv(key)
    for (k, v) in env.items():
        monkeypatch.setenv(k, v)
    assert crashdialog._get_environment_vars() == expected