import pytest
import libqtile.config
from libqtile import layout
from libqtile.confreader import Config
from test.layouts.layout_utils import assert_focus_path, assert_focused

class BspConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a'), libqtile.config.Group('b'), libqtile.config.Group('c'), libqtile.config.Group('d')]
    layouts = [layout.Bsp(), layout.Bsp(margin_on_single=10), layout.Bsp(wrap_clients=True)]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
bsp_config = pytest.mark.parametrize('manager', [BspConfig], indirect=True)

@bsp_config
def test_bsp_window_focus_cycle(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('float1')
    manager.c.window.toggle_floating()
    manager.test_window('float2')
    manager.c.window.toggle_floating()
    manager.test_window('three')
    assert manager.c.layout.info()['clients'] == ['one', 'three', 'two']
    assert_focused(manager, 'three')
    assert_focus_path(manager, 'two', 'float1', 'float2', 'one', 'three')

@bsp_config
def test_bsp_margin_on_single(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    info = manager.c.window.info()
    assert info['x'] == 0
    assert info['y'] == 0
    manager.c.next_layout()
    info = manager.c.window.info()
    assert info['x'] == 10
    assert info['y'] == 10
    manager.test_window('two')
    info = manager.c.window.info()
    assert info['x'] == 0

@bsp_config
def test_bsp_wrap_clients(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    manager.test_window('two')
    assert_focused(manager, 'two')
    manager.c.layout.next()
    assert_focused(manager, 'two')
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    manager.c.next_layout()
    manager.c.next_layout()
    assert_focused(manager, 'one')
    manager.c.layout.next()
    assert_focused(manager, 'two')
    manager.c.layout.next()
    assert_focused(manager, 'one')
    manager.c.layout.previous()
    assert_focused(manager, 'two')