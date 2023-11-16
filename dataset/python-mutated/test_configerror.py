import pytest
import libqtile.bar
import libqtile.config
from libqtile.widget.base import _Widget

class BadWidget(_Widget):

    def _configure(self, qtile, bar):
        if False:
            for i in range(10):
                print('nop')
        _Widget._configure(qtile, bar)
        1 / 0

    def draw(self):
        if False:
            return 10
        pass

@pytest.mark.parametrize('position', ['top', 'bottom', 'left', 'right'])
def test_configerrorwidget(manager_nospawn, minimal_conf_noscreen, position):
    if False:
        i = 10
        return i + 15
    'ConfigError widget should show in any bar orientation.'
    widget = BadWidget(length=10)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(**{position: libqtile.bar.Bar([widget], 10)})]
    manager_nospawn.start(config)
    testbar = manager_nospawn.c.bar[position]
    w = testbar.info()['widgets'][0]
    assert w['name'] == 'configerrorwidget'
    assert w['text'] == 'Widget crashed: BadWidget (click to hide)'
    testbar.fake_button_press(0, position, 0, 0, button=1)
    w = testbar.info()['widgets'][0]
    assert w['text'] == ''