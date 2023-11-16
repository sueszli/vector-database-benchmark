import pytest
import libqtile.config
from libqtile.widget import Systray, TextBox, WidgetBox
from test.widgets.conftest import FakeBar

def test_widgetbox_widget(fake_qtile, fake_window):
    if False:
        i = 10
        return i + 15
    tb_one = TextBox(name='tb_one', text='TB ONE')
    tb_two = TextBox(name='tb_two', text='TB TWO')
    widget_box = WidgetBox(widgets=[tb_one, tb_two], close_button_location='middle', fontsize=10)
    fakebar = FakeBar([widget_box], window=fake_window)
    widget_box._configure(fake_qtile, fakebar)
    assert widget_box.close_button_location == 'left'
    assert fakebar.widgets == [widget_box]
    widget_box.toggle()
    assert widget_box.box_is_open
    assert fakebar.widgets == [widget_box, tb_one, tb_two]
    widget_box.toggle()
    assert not widget_box.box_is_open
    assert fakebar.widgets == [widget_box]
    widget_box.close_button_location = 'right'
    widget_box.toggle()
    assert fakebar.widgets == [tb_one, tb_two, widget_box]

def test_widgetbox_start_opened(manager_nospawn, minimal_conf_noscreen):
    if False:
        return 10
    config = minimal_conf_noscreen
    tbox = TextBox(text='Text Box')
    widget_box = WidgetBox(widgets=[tbox], start_opened=True)
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([widget_box], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    widgets = [w['name'] for w in topbar.info()['widgets']]
    assert widgets == ['widgetbox', 'textbox']

def test_widgetbox_mirror(manager_nospawn, minimal_conf_noscreen):
    if False:
        return 10
    config = minimal_conf_noscreen
    tbox = TextBox(text='Text Box')
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([tbox, WidgetBox(widgets=[tbox])], 10))]
    manager_nospawn.start(config)
    manager_nospawn.c.widget['widgetbox'].toggle()
    topbar = manager_nospawn.c.bar['top']
    widgets = [w['name'] for w in topbar.info()['widgets']]
    assert widgets == ['textbox', 'widgetbox', 'mirror']

def test_widgetbox_mouse_click(manager_nospawn, minimal_conf_noscreen):
    if False:
        print('Hello World!')
    config = minimal_conf_noscreen
    tbox = TextBox(text='Text Box')
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([WidgetBox(widgets=[tbox])], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    assert len(topbar.info()['widgets']) == 1
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    assert len(topbar.info()['widgets']) == 2
    topbar.fake_button_press(0, 'top', 0, 0, button=1)
    assert len(topbar.info()['widgets']) == 1

def test_widgetbox_with_systray_reconfigure_screens_box_open(manager_nospawn, minimal_conf_noscreen, backend_name):
    if False:
        for i in range(10):
            print('nop')
    'Check that Systray does not crash when inside an open widgetbox.'
    if backend_name == 'wayland':
        pytest.skip('Skipping test on Wayland.')
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([WidgetBox(widgets=[Systray()])], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    assert len(topbar.info()['widgets']) == 1
    manager_nospawn.c.widget['widgetbox'].toggle()
    assert len(topbar.info()['widgets']) == 2
    manager_nospawn.c.reconfigure_screens()
    assert len(topbar.info()['widgets']) == 2
    names = [w['name'] for w in topbar.info()['widgets']]
    assert names == ['widgetbox', 'systray']

def test_widgetbox_with_systray_reconfigure_screens_box_closed(manager_nospawn, minimal_conf_noscreen, backend_name):
    if False:
        i = 10
        return i + 15
    'Check that Systray does not crash when inside a closed widgetbox.'
    if backend_name == 'wayland':
        pytest.skip('Skipping test on Wayland.')
    config = minimal_conf_noscreen
    config.screens = [libqtile.config.Screen(top=libqtile.bar.Bar([WidgetBox(widgets=[Systray()])], 10))]
    manager_nospawn.start(config)
    topbar = manager_nospawn.c.bar['top']
    assert len(topbar.info()['widgets']) == 1
    manager_nospawn.c.reconfigure_screens()
    assert len(topbar.info()['widgets']) == 1
    (_, name) = manager_nospawn.c.widget['widgetbox'].eval('self.widgets[0].name')
    assert name == 'systray'

def test_deprecated_configuration(caplog):
    if False:
        i = 10
        return i + 15
    tray = Systray()
    box = WidgetBox([tray])
    assert box.widgets == [tray]
    assert 'The use of a positional argument in WidgetBox is deprecated.' in caplog.text