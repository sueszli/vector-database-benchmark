import pytest
from libqtile.widget import windowtabs

@pytest.fixture
def widget():
    if False:
        return 10
    yield windowtabs.WindowTabs

def ss_window_count(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.test_window('Window One')
    screenshot_manager.test_window('Window Two')
    screenshot_manager.take_screenshot()