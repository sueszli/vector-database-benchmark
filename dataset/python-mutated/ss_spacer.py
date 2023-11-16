import pytest
from libqtile.widget import Spacer

@pytest.fixture
def widget():
    if False:
        i = 10
        return i + 15
    yield Spacer

@pytest.mark.parametrize('screenshot_manager', [{}, {'length': 50}], indirect=True)
def ss_spacer(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()