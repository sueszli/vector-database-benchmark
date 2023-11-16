import pytest
import libqtile.widget

@pytest.fixture
def widget():
    if False:
        while True:
            i = 10
    yield libqtile.widget.CurrentScreen

def ss_currentscreen(screenshot_manager):
    if False:
        return 10
    screenshot_manager.take_screenshot()
    screenshot_manager.c.to_screen(1)
    screenshot_manager.take_screenshot()