import subprocess
from typing import List
import pytest
from libqtile.widget import caps_num_lock_indicator
from test.widgets.conftest import FakeBar

class MockCapsNumLockIndicator:
    CalledProcessError = None
    info: List[List[str]] = []
    is_error = False
    index = 0

    @classmethod
    def reset(cls):
        if False:
            i = 10
            return i + 15
        cls.info = [['Keyboard Control:', '  auto repeat:  on    key click percent:  0    LED mask:  00000002', '  XKB indicators:', '    00: Caps Lock:   off    01: Num Lock:    on     02: Scroll Lock: off', '    03: Compose:     off    04: Kana:        off    05: Sleep:       off'], ['Keyboard Control:', '  auto repeat:  on    key click percent:  0    LED mask:  00000002', '  XKB indicators:', '    00: Caps Lock:   on     01: Num Lock:    on     02: Scroll Lock: off', '    03: Compose:     off    04: Kana:        off    05: Sleep:       off']]
        cls.index = 0
        cls.is_error = False

    @classmethod
    def call_process(cls, cmd):
        if False:
            i = 10
            return i + 15
        if cls.is_error:
            raise subprocess.CalledProcessError(-1, cmd=cmd, output="Couldn't call xset.")
        if cmd[1:] == ['q']:
            track = cls.info[cls.index]
            output = '\n'.join(track)
            return output

def no_op(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    pass

@pytest.fixture
def patched_cnli(monkeypatch):
    if False:
        print('Hello World!')
    MockCapsNumLockIndicator.reset()
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.subprocess', MockCapsNumLockIndicator)
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.subprocess.CalledProcessError', subprocess.CalledProcessError)
    monkeypatch.setattr('libqtile.widget.caps_num_lock_indicator.base.ThreadPoolText.call_process', MockCapsNumLockIndicator.call_process)
    return caps_num_lock_indicator

def test_cnli(fake_qtile, patched_cnli, fake_window):
    if False:
        i = 10
        return i + 15
    widget = patched_cnli.CapsNumLockIndicator()
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == 'Caps off Num on'

def test_cnli_caps_on(fake_qtile, patched_cnli, fake_window):
    if False:
        for i in range(10):
            print('nop')
    widget = patched_cnli.CapsNumLockIndicator()
    MockCapsNumLockIndicator.index = 1
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == 'Caps on Num on'

def test_cnli_error_handling(fake_qtile, patched_cnli, fake_window):
    if False:
        i = 10
        return i + 15
    widget = patched_cnli.CapsNumLockIndicator()
    MockCapsNumLockIndicator.is_error = True
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    text = widget.poll()
    assert text == ''