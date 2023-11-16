from __future__ import annotations
import curses
import importlib
import io
import pytest
import sys
import ansible.utils.display
assert ansible.utils.display
builtin_import = 'builtins.__import__'

def test_pause_curses_tigetstr_none(mocker, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.delitem(sys.modules, 'ansible.utils.display')
    dunder_import = __import__

    def _import(*args, **kwargs):
        if False:
            while True:
                i = 10
        if args[0] == 'curses':
            mock_curses = mocker.Mock()
            mock_curses.setupterm = mocker.Mock(return_value=True)
            mock_curses.tigetstr = mocker.Mock(return_value=None)
            return mock_curses
        else:
            return dunder_import(*args, **kwargs)
    mocker.patch(builtin_import, _import)
    mod = importlib.import_module('ansible.utils.display')
    assert mod.HAS_CURSES is True
    mod.setupterm()
    assert mod.HAS_CURSES is True
    assert mod.MOVE_TO_BOL == b'\r'
    assert mod.CLEAR_TO_EOL == b'\x1b[K'

def test_pause_missing_curses(mocker, monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.delitem(sys.modules, 'ansible.utils.display')
    dunder_import = __import__

    def _import(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if args[0] == 'curses':
            raise ImportError
        else:
            return dunder_import(*args, **kwargs)
    mocker.patch(builtin_import, _import)
    mod = importlib.import_module('ansible.utils.display')
    assert mod.HAS_CURSES is False
    with pytest.raises(AttributeError):
        assert mod.curses
    assert mod.HAS_CURSES is False
    assert mod.MOVE_TO_BOL == b'\r'
    assert mod.CLEAR_TO_EOL == b'\x1b[K'

@pytest.mark.parametrize('exc', (curses.error, TypeError, io.UnsupportedOperation))
def test_pause_curses_setupterm_error(mocker, monkeypatch, exc):
    if False:
        return 10
    monkeypatch.delitem(sys.modules, 'ansible.utils.display')
    dunder_import = __import__

    def _import(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if args[0] == 'curses':
            mock_curses = mocker.Mock()
            mock_curses.setupterm = mocker.Mock(side_effect=exc)
            mock_curses.error = curses.error
            return mock_curses
        else:
            return dunder_import(*args, **kwargs)
    mocker.patch(builtin_import, _import)
    mod = importlib.import_module('ansible.utils.display')
    assert mod.HAS_CURSES is True
    mod.setupterm()
    assert mod.HAS_CURSES is False
    assert mod.MOVE_TO_BOL == b'\r'
    assert mod.CLEAR_TO_EOL == b'\x1b[K'