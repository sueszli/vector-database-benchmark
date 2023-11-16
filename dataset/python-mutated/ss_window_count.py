import pytest
from libqtile.widget import window_count

@pytest.fixture
def widget():
    if False:
        print('Hello World!')
    yield window_count.WindowCount

def ss_window_count(screenshot_manager):
    if False:
        return 10
    screenshot_manager.test_window('One')
    screenshot_manager.test_window('Two')
    screenshot_manager.take_screenshot()