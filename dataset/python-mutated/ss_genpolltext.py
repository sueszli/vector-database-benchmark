import pytest
import libqtile.widget

@pytest.fixture
def widget():
    if False:
        while True:
            i = 10
    yield libqtile.widget.GenPollText

@pytest.mark.parametrize('screenshot_manager', [{'func': lambda : 'Function text.'}], indirect=True)
def ss_genpolltext(screenshot_manager):
    if False:
        print('Hello World!')
    screenshot_manager.take_screenshot()