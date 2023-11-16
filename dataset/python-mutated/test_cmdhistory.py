"""Tests for misc.cmdhistory.History."""
import unittest.mock
import pytest
from qutebrowser.misc import cmdhistory
from qutebrowser.utils import objreg
HISTORY = ['first', 'second', 'third', 'fourth', 'fifth']

@pytest.fixture
def hist():
    if False:
        return 10
    return cmdhistory.History(history=HISTORY)

def test_no_history():
    if False:
        print('Hello World!')
    hist = cmdhistory.History()
    assert hist.history == []

def test_history():
    if False:
        i = 10
        return i + 15
    hist = cmdhistory.History(history=HISTORY)
    assert hist.history == HISTORY

@pytest.mark.parametrize('tmphist, expected', [(None, False), (HISTORY, True)])
def test_is_browsing(hist, tmphist, expected):
    if False:
        print('Hello World!')
    hist._tmphist = tmphist
    assert hist.is_browsing() == expected

def test_start_stop(hist):
    if False:
        while True:
            i = 10
    assert not hist.is_browsing()
    hist.start('s')
    assert hist.is_browsing()
    hist.stop()
    assert not hist.is_browsing()

def test_start_with_text(hist):
    if False:
        while True:
            i = 10
    "Test start with given 'text'."
    hist.start('f')
    assert 'first' in hist._tmphist
    assert 'fourth' in hist._tmphist
    assert 'second' not in hist._tmphist

def test_start_no_text(hist):
    if False:
        print('Hello World!')
    'Test start with no given text.'
    hist.start('')
    assert list(hist._tmphist) == HISTORY

def test_start_no_items(hist):
    if False:
        for i in range(10):
            print('nop')
    'Test start with no matching text.'
    with pytest.raises(cmdhistory.HistoryEmptyError):
        hist.start('k')
    assert not hist._tmphist

def test_getitem(hist):
    if False:
        return 10
    'Test __getitem__.'
    assert hist[0] == HISTORY[0]

def test_setitem(hist):
    if False:
        return 10
    'Test __setitem__.'
    with pytest.raises(TypeError, match="'History' object does not support item assignment"):
        hist[0] = 'foo'

def test_not_browsing_error(hist):
    if False:
        print('Hello World!')
    'Test that next/previtem throws a ValueError.'
    with pytest.raises(ValueError, match='Currently not browsing history'):
        hist.nextitem()
    with pytest.raises(ValueError, match='Currently not browsing history'):
        hist.previtem()

def test_nextitem_single(hist, monkeypatch):
    if False:
        i = 10
        return i + 15
    'Test nextitem() with valid input.'
    hist.start('f')
    monkeypatch.setattr(hist._tmphist, 'nextitem', lambda : 'item')
    assert hist.nextitem() == 'item'

def test_previtem_single(hist, monkeypatch):
    if False:
        while True:
            i = 10
    'Test previtem() with valid input.'
    hist.start('f')
    monkeypatch.setattr(hist._tmphist, 'previtem', lambda : 'item')
    assert hist.previtem() == 'item'

def test_nextitem_previtem_chain(hist):
    if False:
        return 10
    'Test a combination of nextitem and previtem statements.'
    assert hist.start('f') == 'fifth'
    assert hist.previtem() == 'fourth'
    assert hist.previtem() == 'first'
    assert hist.nextitem() == 'fourth'

def test_nextitem_index_error(hist):
    if False:
        i = 10
        return i + 15
    'Test nextitem() when _tmphist raises an IndexError.'
    hist.start('f')
    with pytest.raises(cmdhistory.HistoryEndReachedError):
        hist.nextitem()

def test_previtem_index_error(hist):
    if False:
        i = 10
        return i + 15
    'Test previtem() when _tmphist raises an IndexError.'
    hist.start('f')
    with pytest.raises(cmdhistory.HistoryEndReachedError):
        for _ in range(10):
            hist.previtem()

def test_append_private_mode(hist, config_stub):
    if False:
        for i in range(10):
            print('nop')
    'Test append in private mode.'
    hist._private = True
    config_stub.val.content.private_browsing = True
    hist.append('new item')
    assert hist.history == HISTORY

def test_append(hist):
    if False:
        for i in range(10):
            print('nop')
    'Test append outside private mode.'
    hist.append('new item')
    assert 'new item' in hist.history
    hist.history.remove('new item')
    assert hist.history == HISTORY

def test_append_empty_history(hist):
    if False:
        print('Hello World!')
    'Test append when .history is empty.'
    hist.history = []
    hist.append('item')
    assert hist[0] == 'item'

def test_append_double(hist):
    if False:
        i = 10
        return i + 15
    hist.append('fifth')
    assert hist.history[-2:] == ['fourth', 'fifth']

@pytest.fixture
def init_patch():
    if False:
        i = 10
        return i + 15
    yield
    objreg.delete('command-history')

def test_init(init_patch, fake_save_manager, data_tmpdir, config_stub):
    if False:
        i = 10
        return i + 15
    cmdhistory.init()
    fake_save_manager.add_saveable.assert_any_call('command-history', unittest.mock.ANY, unittest.mock.ANY)