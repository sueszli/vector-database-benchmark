import pytest
import libqtile.bar
import libqtile.config
import libqtile.confreader
import libqtile.layout
from libqtile.command.interface import CommandException
from libqtile.widget.crashme import _CrashMe

def test_crashme_init(manager_nospawn, minimal_conf_noscreen):
    if False:
        for i in range(10):
            print('nop')
    crash = _CrashMe()
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([crash], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    w = topbar.info()['widgets'][0]
    assert w['name'] == '_crashme'
    assert w['text'] == 'Crash me !'
    with pytest.raises(CommandException) as e_info:
        topbar.fake_button_press(0, 'top', 0, 0, button=1)
    assert e_info.match('ZeroDivisionError')
    with pytest.raises(CommandException) as e_info:
        topbar.fake_button_press(0, 'top', 0, 0, button=3)
    assert e_info.match('parse_markup[(][)] failed')