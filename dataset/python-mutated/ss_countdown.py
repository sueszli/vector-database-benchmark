from datetime import datetime, timedelta
import pytest
import libqtile.widget
td = timedelta(days=1, hours=2, minutes=34, seconds=56)

@pytest.fixture
def widget():
    if False:
        while True:
            i = 10
    yield libqtile.widget.Countdown

@pytest.mark.parametrize('screenshot_manager', [{'date': datetime.now() + td}], indirect=True)
def ss_countdown(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.take_screenshot()