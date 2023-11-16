from datetime import timedelta
import pytest
from libqtile.widget import pomodoro
from test.widgets.test_pomodoro import MockDatetime

def increment_time(self, increment):
    if False:
        i = 10
        return i + 15
    MockDatetime._adjustment += timedelta(minutes=increment)

@pytest.fixture
def widget(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr('libqtile.widget.pomodoro.datetime', MockDatetime)
    pomodoro.Pomodoro.adjust_time = increment_time
    yield pomodoro.Pomodoro

def ss_pomodoro(screenshot_manager):
    if False:
        print('Hello World!')
    bar = screenshot_manager.c.bar['top']
    widget = screenshot_manager.c.widget['pomodoro']
    screenshot_manager.take_screenshot()
    bar.fake_button_press(0, 'top', 0, 0, 3)
    widget.eval('self.update(self.poll())')
    screenshot_manager.take_screenshot()
    widget.eval('self.adjust_time(25)')
    widget.eval('self.update(self.poll())')
    screenshot_manager.take_screenshot()