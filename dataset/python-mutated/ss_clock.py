import pytest
from libqtile.widget import Clock

@pytest.fixture
def widget():
    if False:
        for i in range(10):
            print('nop')
    yield Clock

@pytest.mark.parametrize('screenshot_manager', [{}, {'format': '%d/%m/%y %H:%M'}], indirect=True)
def ss_clock(screenshot_manager):
    if False:
        for i in range(10):
            print('nop')
    screenshot_manager.take_screenshot()