import sys
from importlib import reload
import pytest
from test.widgets.test_keyboardkbdd import Mockconstants, MockSpawn, mock_signal_receiver

@pytest.fixture
def widget(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setitem(sys.modules, 'dbus_next.constants', Mockconstants('dbus_next.constants'))
    from libqtile.widget import keyboardkbdd
    reload(keyboardkbdd)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.MessageType', Mockconstants.MessageType)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.KeyboardKbdd.call_process', MockSpawn.call_process)
    monkeypatch.setattr('libqtile.widget.keyboardkbdd.add_signal_receiver', mock_signal_receiver)
    return keyboardkbdd.KeyboardKbdd

@pytest.mark.parametrize('screenshot_manager', [{'configured_keyboards': ['gb', 'us']}], indirect=True)
def ss_keyboardkbdd(screenshot_manager):
    if False:
        return 10
    screenshot_manager.take_screenshot()