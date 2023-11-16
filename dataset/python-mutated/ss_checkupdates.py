import pytest
import libqtile.widget
from test.widgets.test_check_updates import MockPopen, MockSpawn

@pytest.fixture
def widget(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.setattr('libqtile.widget.base.subprocess.check_output', MockSpawn.call_process)
    monkeypatch.setattr('libqtile.widget.check_updates.Popen', MockPopen)
    yield libqtile.widget.CheckUpdates

@pytest.mark.parametrize('screenshot_manager', [{'no_update_string': 'No updates'}], indirect=True)
def ss_checkupdates(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()
    screenshot_manager.c.widget['checkupdates'].eval('self.update(self.poll())')
    screenshot_manager.take_screenshot()