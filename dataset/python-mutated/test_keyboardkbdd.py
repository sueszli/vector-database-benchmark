import sys
from importlib import reload
from types import ModuleType
import pytest
from test.widgets.conftest import FakeBar

async def mock_signal_receiver(*args, **kwargs):
    return True

class Mockconstants(ModuleType):

    class MessageType:
        SIGNAL = 1

class MockSpawn:
    call_count = 0

    @classmethod
    def call_process(cls, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if cls.call_count == 0:
            cls.call_count += 1
            return ''
        return 'kbdd'

class MockMessage:

    def __init__(self, is_signal=True, body=0):
        if False:
            return 10
        self.message_type = 1 if is_signal else 0
        self.body = [body]

@pytest.fixture
def patched_widget(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setitem(sys.modules, 'dbus_next.constants', Mockconstants('dbus_next.constants'))
    from libqtile.widget import keyboardkbdd
    reload(keyboardkbdd)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.MessageType', Mockconstants.MessageType)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.KeyboardKbdd.call_process', MockSpawn.call_process)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.add_signal_receiver', mock_signal_receiver)
    return keyboardkbdd

def test_keyboardkbdd_process_running(fake_qtile, patched_widget, fake_window):
    if False:
        while True:
            i = 10
    MockSpawn.call_count = 1
    kbd = patched_widget.KeyboardKbdd(configured_keyboards=['gb', 'us'])
    fakebar = FakeBar([kbd], window=fake_window)
    kbd._configure(fake_qtile, fakebar)
    assert kbd.is_kbdd_running
    assert kbd.keyboard == 'gb'
    message = MockMessage(body=1)
    kbd._signal_received(message)
    assert kbd.keyboard == 'us'
    message = MockMessage(body=0, is_signal=False)
    kbd._signal_received(message)
    assert kbd.keyboard == 'us'

def test_keyboardkbdd_process_not_running(fake_qtile, patched_widget, fake_window):
    if False:
        while True:
            i = 10
    MockSpawn.call_count = 0
    kbd = patched_widget.KeyboardKbdd(configured_keyboards=['gb', 'us'])
    fakebar = FakeBar([kbd], window=fake_window)
    kbd._configure(fake_qtile, fakebar)
    assert not kbd.is_kbdd_running
    assert kbd.keyboard == 'N/A'
    kbd.poll()
    assert kbd.keyboard == 'gb'

def test_keyboard_kbdd_colours(fake_qtile, patched_widget, fake_window):
    if False:
        print('Hello World!')
    MockSpawn.call_count = 1
    kbd = patched_widget.KeyboardKbdd(configured_keyboards=['gb', 'us'], colours=['#ff0000', '#00ff00'])
    fakebar = FakeBar([kbd], window=fake_window)
    kbd._configure(fake_qtile, fakebar)
    message = MockMessage(body=0)
    kbd._signal_received(message)
    assert kbd.layout.colour == '#ff0000'
    message = MockMessage(body=1)
    kbd._signal_received(message)
    assert kbd.layout.colour == '#00ff00'
    kbd.colours = '#ffff00'
    kbd._set_colour(1)
    assert kbd.layout.colour == '#00ff00'
    kbd.colours = ['#ff00ff']
    kbd._set_colour(1)
    assert kbd.layout.colour == '#ff00ff'