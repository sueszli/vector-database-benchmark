from datetime import datetime, timedelta
from importlib import reload
import pytest
from libqtile.widget import pomodoro
from test.widgets.conftest import FakeBar
COLOR_INACTIVE = '123456'
COLOR_ACTIVE = '654321'
COLOR_BREAK = 'AABBCC'
PREFIX_INACTIVE = 'TESTING POMODORO'
PREFIX_ACTIVE = 'ACTIVE'
PREFIX_BREAK = 'BREAK'
PREFIX_LONG_BREAK = 'LONG BREAK'
PREFIX_PAUSED = 'PAUSING'

class MockDatetime(datetime):
    _adjustment = timedelta(0)

    @classmethod
    def now(cls, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return cls(2021, 1, 1, 12, 0, 0) + cls._adjustment

@pytest.fixture
def patched_widget(monkeypatch):
    if False:
        print('Hello World!')
    reload(pomodoro)
    monkeypatch.setattr('libqtile.widget.pomodoro.datetime', MockDatetime)
    yield pomodoro

@pytest.mark.usefixtures('patched_widget')
def test_pomodoro(fake_qtile, fake_window):
    if False:
        print('Hello World!')
    widget = pomodoro.Pomodoro(update_interval=100, color_active=COLOR_ACTIVE, color_inactive=COLOR_INACTIVE, color_break=COLOR_BREAK, num_pomodori=2, length_pomodori=15, length_short_break=5, length_long_break=10, notification_on=False, prefix_inactive=PREFIX_INACTIVE, prefix_active=PREFIX_ACTIVE, prefix_break=PREFIX_BREAK, prefix_long_break=PREFIX_LONG_BREAK, prefix_paused=PREFIX_PAUSED)
    fakebar = FakeBar([widget], window=fake_window)
    widget._configure(fake_qtile, fakebar)
    assert widget.poll() == PREFIX_INACTIVE
    assert widget.layout.colour == COLOR_INACTIVE
    widget.toggle_break()
    assert widget.poll() == f'{PREFIX_ACTIVE}0:15:0'
    assert widget.layout.colour == COLOR_ACTIVE
    widget.toggle_break()
    assert widget.poll() == PREFIX_PAUSED
    assert widget.layout.colour == COLOR_INACTIVE
    widget.toggle_break()
    MockDatetime._adjustment += timedelta(minutes=5)
    assert widget.poll() == f'{PREFIX_ACTIVE}0:10:0'
    assert widget.layout.colour == COLOR_ACTIVE
    MockDatetime._adjustment += timedelta(minutes=10)
    assert widget.poll() == f'{PREFIX_BREAK}0:5:0'
    assert widget.layout.colour == COLOR_BREAK
    MockDatetime._adjustment += timedelta(minutes=5)
    assert widget.poll() == f'{PREFIX_ACTIVE}0:15:0'
    assert widget.layout.colour == COLOR_ACTIVE
    MockDatetime._adjustment += timedelta(minutes=15)
    assert widget.poll() == f'{PREFIX_LONG_BREAK}0:10:0'
    assert widget.layout.colour == COLOR_BREAK
    MockDatetime._adjustment += timedelta(minutes=10)
    assert widget.poll() == f'{PREFIX_ACTIVE}0:15:0'
    MockDatetime._adjustment += timedelta(minutes=10)
    assert widget.poll() == f'{PREFIX_ACTIVE}0:5:0'
    widget.toggle_active()
    assert widget.poll() == PREFIX_INACTIVE
    widget.toggle_active()
    assert widget.poll() == f'{PREFIX_ACTIVE}0:15:0'