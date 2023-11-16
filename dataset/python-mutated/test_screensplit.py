import pytest
import libqtile.config
from libqtile import layout
from libqtile.config import Match
from libqtile.confreader import Config
from test.layouts.layout_utils import assert_dimensions

class ScreenSplitConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.ScreenSplit()]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
screensplit_config = pytest.mark.parametrize('manager', [ScreenSplitConfig], indirect=True)

@screensplit_config
def test_screensplit(manager):
    if False:
        while True:
            i = 10
    assert manager.c.layout.info()['current_layout'] == 'max'
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 800, 300)
    manager.test_window('two')
    assert_dimensions(manager, 0, 0, 800, 300)
    assert manager.c.layout.info()['current_clients'] == ['one', 'two']
    manager.c.layout.next_split()
    assert manager.c.layout.info()['current_layout'] == 'columns'
    assert manager.c.layout.info()['current_clients'] == []
    manager.test_window('three')
    assert_dimensions(manager, 0, 300, 800, 300)
    manager.test_window('four')
    assert_dimensions(manager, 400, 300, 396, 296)
    assert manager.c.layout.info()['current_clients'] == ['three', 'four']
    manager.c.layout.next_split()
    assert manager.c.layout.info()['current_layout'] == 'max'
    assert manager.c.layout.info()['current_clients'] == ['one', 'two']

@screensplit_config
def test_commands_passthrough(manager):
    if False:
        i = 10
        return i + 15
    assert manager.c.layout.info()['current_layout'] == 'max'
    assert 'grow_left' not in manager.c.layout.commands()
    manager.c.layout.next_split()
    assert manager.c.layout.info()['current_layout'] == 'columns'
    manager.test_window('one')
    assert_dimensions(manager, 0, 300, 800, 300)
    manager.test_window('two')
    assert_dimensions(manager, 400, 300, 396, 296)
    assert 'grow_left' in manager.c.layout.commands()
    manager.c.layout.grow_left()
    assert_dimensions(manager, 360, 300, 436, 296)

@screensplit_config
def test_move_window_to_split(manager):
    if False:
        for i in range(10):
            print('nop')
    assert manager.c.layout.info()['current_layout'] == 'max'
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 800, 300)
    manager.c.layout.move_window_to_next_split()
    assert manager.c.layout.info()['current_layout'] == 'columns'
    assert_dimensions(manager, 0, 300, 800, 300)
    manager.c.layout.move_window_to_previous_split()
    assert manager.c.layout.info()['current_layout'] == 'max'
    assert_dimensions(manager, 0, 0, 800, 300)

def test_invalid_splits():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'rect': (0, 0, 1, 1)}])
    assert str(e.value) == "Splits must define 'name', 'rect' and 'layout'."
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': '0, 0, 1, 1', 'layout': layout.Max()}])
    assert str(e.value) == 'Split rect should be a list/tuple.'
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1), 'layout': layout.Max()}])
    assert str(e.value) == 'Split rect should have 4 float/int members.'
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, '1'), 'layout': layout.Max()}])
    assert str(e.value) == 'Split rect should have 4 float/int members.'
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, 1), 'layout': layout.ScreenSplit()}])
    assert str(e.value) == 'ScreenSplit layouts cannot be nested.'
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, 1), 'layout': layout.Max(), 'matches': [True]}])
    assert str(e.value) == "Invalid object in 'matches'."
    with pytest.raises(ValueError) as e:
        layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, 1), 'layout': layout.Max(), 'matches': Match(wm_class='test')}])
    assert str(e.value) == "'matches' must be a list of 'Match' objects."
    s_split = layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, 1), 'layout': layout.Max()}])
    assert s_split
    s_split = layout.ScreenSplit(splits=[{'name': 'test', 'rect': (0, 0, 1, 1), 'layout': layout.Max(), 'matches': [Match(wm_class='test')]}])
    assert s_split