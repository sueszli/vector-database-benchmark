import sys
from importlib import reload
from types import ModuleType
import pytest
from test.widgets.conftest import FakeBar

def no_op(*args, **kwargs):
    if False:
        print('Hello World!')
    pass

async def mock_signal_receiver(*args, **kwargs):
    return True

def fake_timer(interval, func, *args, **kwargs):
    if False:
        print('Hello World!')

    class TimerObj:

        def cancel(self):
            if False:
                return 10
            pass

        @property
        def _scheduled(self):
            if False:
                while True:
                    i = 10
            return False
    return TimerObj()

class MockConstants(ModuleType):

    class MessageType:
        SIGNAL = 1

class MockMessage:

    def __init__(self, is_signal=True, body=None):
        if False:
            i = 10
            return i + 15
        self.message_type = 1 if is_signal else 0
        self.body = body

class obj:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value

def metadata_and_status(status):
    if False:
        while True:
            i = 10
    return MockMessage(body=('', {'Metadata': obj({'mpris:trackid': obj(1), 'xesam:url': obj('/path/to/rickroll.mp3'), 'xesam:title': obj('Never Gonna Give You Up'), 'xesam:artist': obj(['Rick Astley']), 'xesam:album': obj('Whenever You Need Somebody'), 'mpris:length': obj(200000000)}), 'PlaybackStatus': obj(status)}, []))

def playback_status(status, signal=True):
    if False:
        return 10
    return MockMessage(is_signal=signal, body=('', {'PlaybackStatus': obj(status)}, []))
METADATA_PLAYING = metadata_and_status('Playing')
METADATA_PAUSED = metadata_and_status('Paused')
STATUS_PLAYING = playback_status('Playing')
STATUS_PAUSED = playback_status('Paused')
STATUS_STOPPED = playback_status('Stopped')
NON_SIGNAL = playback_status('Paused', False)

@pytest.fixture
def patched_module(monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.delitem(sys.modules, 'dbus_next.constants', raising=False)
    monkeypatch.setitem(sys.modules, 'dbus_next.constants', MockConstants('dbus_next.constants'))
    from libqtile.widget import mpris2widget
    reload(mpris2widget)
    monkeypatch.setattr('libqtile.widget.mpris2widget.add_signal_receiver', mock_signal_receiver)
    return mpris2widget

def test_mpris2_signal_handling(fake_qtile, patched_module, fake_window):
    if False:
        for i in range(10):
            print('nop')
    mp = patched_module.Mpris2()
    fakebar = FakeBar([mp], window=fake_window)
    mp.timeout_add = fake_timer
    mp._configure(fake_qtile, fakebar)
    assert mp.displaytext == ''
    mp.parse_message(*METADATA_PLAYING.body)
    assert mp.displaytext == ''
    mp.configured = True
    mp.parse_message(*METADATA_PLAYING.body)
    assert mp.text == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*STATUS_PAUSED.body)
    assert mp.text == 'Paused: Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*STATUS_STOPPED.body)
    assert mp.displaytext == ''
    mp.parse_message(*METADATA_PLAYING.body)
    assert mp.text == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*METADATA_PAUSED.body)
    assert mp.text == 'Paused: Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*STATUS_PLAYING.body)
    assert mp.text == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    info = mp.info()
    assert info['text'] == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    assert info['isplaying']

def test_mpris2_custom_stop_text(fake_qtile, patched_module, fake_window):
    if False:
        return 10
    mp = patched_module.Mpris2(stop_pause_text='Test Paused')
    fakebar = FakeBar([mp], window=fake_window)
    mp.timeout_add = fake_timer
    mp._configure(fake_qtile, fakebar)
    mp.configured = True
    mp.parse_message(*METADATA_PLAYING.body)
    assert mp.text == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*STATUS_PAUSED.body)
    assert mp.text == 'Test Paused'

def test_mpris2_no_metadata(fake_qtile, patched_module, fake_window):
    if False:
        return 10
    mp = patched_module.Mpris2()
    fakebar = FakeBar([mp], window=fake_window)
    mp.timeout_add = fake_timer
    mp._configure(fake_qtile, fakebar)
    mp.configured = True
    mp.parse_message(*STATUS_PLAYING.body)
    assert mp.text == 'No metadata for current track'

def test_mpris2_no_scroll(fake_qtile, patched_module, fake_window):
    if False:
        while True:
            i = 10
    mp = patched_module.Mpris2(scroll_chars=None)
    fakebar = FakeBar([mp], window=fake_window)
    mp.timeout_add = fake_timer
    mp._configure(fake_qtile, fakebar)
    mp.configured = True
    mp.parse_message(*METADATA_PLAYING.body)
    assert mp.text == 'Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'
    mp.parse_message(*METADATA_PAUSED.body)
    assert mp.text == 'Paused: Never Gonna Give You Up - Whenever You Need Somebody - Rick Astley'

def test_mpris2_deprecated_format(patched_module):
    if False:
        i = 10
        return i + 15
    '\n    Previously, metadata was displayed by using a list of fields.\n    Now, we use a `format` string. The widget should create this when a user\n    provides `display_metadata` in their config.\n    '
    mp = patched_module.Mpris2(display_metadata=['xesam:title', 'xesam:artist'])
    assert mp.format == '{xesam:title} - {xesam:artist}'