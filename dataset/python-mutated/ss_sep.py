import pytest
from libqtile.widget import Sep

@pytest.fixture
def widget():
    if False:
        print('Hello World!')
    yield Sep

@pytest.mark.parametrize('screenshot_manager', [{}, {'padding': 10, 'linewidth': 5, 'size_percent': 50}], indirect=True)
def ss_sep(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.take_screenshot()