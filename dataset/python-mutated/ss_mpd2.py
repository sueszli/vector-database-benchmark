import sys
import pytest
import libqtile.widget
from test.widgets.test_mpd2widget import MockMPD

@pytest.fixture
def widget(monkeypatch):
    if False:
        return 10
    monkeypatch.setitem(sys.modules, 'mpd', MockMPD('mpd'))
    yield libqtile.widget.Mpd2

@pytest.mark.parametrize('screenshot_manager', [{}, {'status_format': '{play_status} {artist}/{title}'}], indirect=True)
def ss_mpd2(screenshot_manager):
    if False:
        for i in range(10):
            print('nop')
    screenshot_manager.take_screenshot()

@pytest.mark.parametrize('screenshot_manager', [{'idle_format': '{play_status} {idle_message}', 'idle_message': 'MPD not playing'}], indirect=True)
def ss_mpd2_idle(screenshot_manager):
    if False:
        while True:
            i = 10
    widget = screenshot_manager.c.widget['mpd2']
    widget.eval('self.client.force_idle()')
    widget.eval('self.update(self.poll())')
    widget.eval('self.bar.draw()')
    screenshot_manager.take_screenshot()