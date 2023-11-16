import pytest
import libqtile.bar
import libqtile.config
import libqtile.confreader
import libqtile.layout
from libqtile import widget
from libqtile.ipc import IPCError

def test_trigger_and_cancel(manager_nospawn, minimal_conf_noscreen):
    if False:
        while True:
            i = 10
    qewidget = widget.QuickExit(timer_interval=100)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([qewidget], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    w = topbar.info()['widgets'][0]
    assert w['text'] == '[ shutdown ]'
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    w = topbar.info()['widgets'][0]
    assert w['text'] == '[ 4 seconds ]'
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    w = topbar.info()['widgets'][0]
    assert w['text'] == '[ shutdown ]'

def test_exit(manager_nospawn, minimal_conf_noscreen):
    if False:
        for i in range(10):
            print('nop')
    qewidget = widget.QuickExit(timer_interval=0.001, countdown_start=1)
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([qewidget], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    with pytest.raises((IPCError, ConnectionResetError)):
        assert topbar.info()