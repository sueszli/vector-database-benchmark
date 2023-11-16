import libqtile.bar
import libqtile.config
import libqtile.confreader
import libqtile.layout
from libqtile import widget
from test.conftest import dualmonitor
ACTIVE = '#FF0000'
INACTIVE = '#00FF00'

@dualmonitor
def test_change_screen(manager_nospawn, minimal_conf_noscreen):
    if False:
        print('Hello World!')
    cswidget = widget.CurrentScreen(active_color=ACTIVE, inactive_color=INACTIVE)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([cswidget], 10)), libqtile.config.Screen()]
    manager_nospawn.start(config)
    w = manager_nospawn.c.screen[0].bar['top'].info()['widgets'][0]
    assert w['text'] == 'A'
    assert w['foreground'] == ACTIVE
    manager_nospawn.c.to_screen(1)
    w = manager_nospawn.c.screen[0].bar['top'].info()['widgets'][0]
    assert w['text'] == 'I'
    assert w['foreground'] == INACTIVE