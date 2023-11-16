import pytest
from libqtile.widget import QuickExit

@pytest.fixture
def widget():
    if False:
        i = 10
        return i + 15
    yield QuickExit

@pytest.mark.parametrize('screenshot_manager', [{}, {'default_text': '[X]', 'countdown_format': '[{}]'}], indirect=True)
def ss_quickexit(screenshot_manager):
    if False:
        return 10
    screenshot_manager.take_screenshot()
    screenshot_manager.c.bar['top'].fake_button_press(0, 'top', 0, 0, button=1)
    screenshot_manager.take_screenshot()