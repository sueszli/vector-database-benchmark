from functools import partial
import pytest
from libqtile.widget import TextBox

@pytest.fixture
def widget():
    if False:
        i = 10
        return i + 15
    yield partial(TextBox, 'Testing Text Box')

@pytest.mark.parametrize('screenshot_manager', [{}, {'foreground': '2980b9'}], indirect=True)
def ss_text(screenshot_manager):
    if False:
        i = 10
        return i + 15
    screenshot_manager.take_screenshot()