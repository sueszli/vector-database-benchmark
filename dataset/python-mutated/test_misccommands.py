"""Tests for qutebrowser.components.misccommands."""
import signal
import contextlib
import time
import pytest
from qutebrowser.api import cmdutils
from qutebrowser.utils import utils
from qutebrowser.components import misccommands

@contextlib.contextmanager
def _trapped_segv(handler):
    if False:
        print('Hello World!')
    'Temporarily install given signal handler for SIGSEGV.'
    old_handler = signal.signal(signal.SIGSEGV, handler)
    yield
    if old_handler is not None:
        signal.signal(signal.SIGSEGV, old_handler)

def test_debug_crash_exception():
    if False:
        return 10
    'Verify that debug_crash crashes as intended.'
    with pytest.raises(Exception, match='Forced crash'):
        misccommands.debug_crash(typ='exception')

@pytest.mark.skipif(utils.is_windows, reason="current CPython/win can't recover from SIGSEGV")
def test_debug_crash_segfault():
    if False:
        while True:
            i = 10
    'Verify that debug_crash crashes as intended.'
    caught = False

    def _handler(num, frame):
        if False:
            return 10
        'Temporary handler for segfault.'
        nonlocal caught
        caught = num == signal.SIGSEGV
    with _trapped_segv(_handler):
        with pytest.raises(Exception, match='Segfault failed'):
            misccommands.debug_crash(typ='segfault')
        time.sleep(0.001)
    assert caught

def test_debug_trace(mocker):
    if False:
        print('Hello World!')
    'Check if hunter.trace is properly called.'
    pytest.importorskip('hunter')
    hunter_mock = mocker.patch.object(misccommands, 'hunter')
    misccommands.debug_trace(1)
    hunter_mock.trace.assert_called_with(1)

def test_debug_trace_exception(mocker):
    if False:
        while True:
            i = 10
    'Check that exceptions thrown by hunter.trace are handled.'

    def _mock_exception():
        if False:
            return 10
        "Side effect for testing debug_trace's reraise."
        raise Exception('message')
    hunter_mock = mocker.patch.object(misccommands, 'hunter')
    hunter_mock.trace.side_effect = _mock_exception
    with pytest.raises(cmdutils.CommandError, match='Exception: message'):
        misccommands.debug_trace()

def test_debug_trace_no_hunter(monkeypatch):
    if False:
        return 10
    'Test that an error is shown if debug_trace is called without hunter.'
    monkeypatch.setattr(misccommands, 'hunter', None)
    with pytest.raises(cmdutils.CommandError, match="You need to install 'hunter' to use this command!"):
        misccommands.debug_trace()