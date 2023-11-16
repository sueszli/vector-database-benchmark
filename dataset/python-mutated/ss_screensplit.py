import pytest
import libqtile.config
import libqtile.confreader
import libqtile.layout
import libqtile.resources.default_config
from libqtile.widget import ScreenSplit

@pytest.fixture
def widget():
    if False:
        for i in range(10):
            print('nop')
    yield ScreenSplit

@pytest.fixture(scope='function')
def minimal_conf_noscreen():
    if False:
        return 10

    class MinimalConf(libqtile.confreader.Config):
        auto_fullscreen = False
        keys = []
        mouse = []
        groups = [libqtile.config.Group('a'), libqtile.config.Group('b')]
        layouts = [libqtile.layout.ScreenSplit()]
        floating_layout = libqtile.resources.default_config.floating_layout
        screens = []
    return MinimalConf

def ss_screensplit(screenshot_manager):
    if False:
        for i in range(10):
            print('nop')
    screenshot_manager.take_screenshot()