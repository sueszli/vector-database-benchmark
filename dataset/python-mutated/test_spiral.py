import pytest
import libqtile.config
from libqtile import layout
from libqtile.confreader import Config
from test.layouts.layout_utils import assert_dimensions

class SpiralConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.Spiral(ratio=0.5, new_client_position='bottom'), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='top'), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='right'), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='bottom')]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
spiral_config = pytest.mark.parametrize('manager', [SpiralConfig], indirect=True)

class AnticlockwiseConfig(SpiralConfig):
    layouts = [layout.Spiral(ratio=0.5, new_client_position='bottom', clockwise=False), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='top', clockwise=False), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='right', clockwise=False), layout.Spiral(ratio=0.5, new_client_position='bottom', main_pane='bottom', clockwise=False)]
anticlockwise_config = pytest.mark.parametrize('manager', [AnticlockwiseConfig], indirect=True)

@spiral_config
def test_spiral_left(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 400, 0, 398, 598)
    manager.test_window('three')
    assert_dimensions(manager, 400, 300, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 400, 300, 198, 298)
    manager.test_window('five')
    assert_dimensions(manager, 400, 300, 198, 148)

@spiral_config
def test_spiral_top(manager):
    if False:
        while True:
            i = 10
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 300, 798, 298)
    manager.test_window('three')
    assert_dimensions(manager, 0, 300, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 0, 300, 398, 148)
    manager.test_window('five')
    assert_dimensions(manager, 200, 300, 198, 148)

@spiral_config
def test_spiral_right(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.c.next_layout()
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 0, 398, 598)
    manager.test_window('three')
    assert_dimensions(manager, 0, 0, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 200, 0, 198, 298)
    manager.test_window('five')
    assert_dimensions(manager, 200, 150, 198, 148)

@spiral_config
def test_spiral_bottom(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.c.next_layout()
    manager.c.next_layout()
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 0, 798, 298)
    manager.test_window('three')
    assert_dimensions(manager, 400, 0, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 400, 150, 398, 148)
    manager.test_window('five')
    assert_dimensions(manager, 400, 150, 198, 148)

@anticlockwise_config
def test_spiral_left_anticlockwise(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 400, 0, 398, 598)
    manager.test_window('three')
    assert_dimensions(manager, 400, 0, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 400, 0, 198, 298)
    manager.test_window('five')
    assert_dimensions(manager, 400, 150, 198, 148)

@anticlockwise_config
def test_spiral_top_anticlockwise(manager):
    if False:
        print('Hello World!')
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 300, 798, 298)
    manager.test_window('three')
    assert_dimensions(manager, 400, 300, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 400, 300, 398, 148)
    manager.test_window('five')
    assert_dimensions(manager, 400, 300, 198, 148)

@anticlockwise_config
def test_spiral_right_anticlockwise(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.c.next_layout()
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 0, 398, 598)
    manager.test_window('three')
    assert_dimensions(manager, 0, 300, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 200, 300, 198, 298)
    manager.test_window('five')
    assert_dimensions(manager, 200, 300, 198, 148)

@anticlockwise_config
def test_spiral_bottom_anticlockwise(manager):
    if False:
        i = 10
        return i + 15
    manager.c.next_layout()
    manager.c.next_layout()
    manager.c.next_layout()
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 0, 0, 798, 298)
    manager.test_window('three')
    assert_dimensions(manager, 0, 0, 398, 298)
    manager.test_window('four')
    assert_dimensions(manager, 0, 150, 398, 148)
    manager.test_window('five')
    assert_dimensions(manager, 200, 150, 198, 148)

@spiral_config
def test_spiral_adjust_master_ratios(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 400, 0, 398, 598)
    manager.c.layout.grow_main()
    assert_dimensions(manager, 480, 0, 318, 598)
    manager.c.layout.grow_main()
    assert_dimensions(manager, 560, 0, 238, 598)
    for _ in range(4):
        manager.c.layout.shrink_main()
    assert_dimensions(manager, 240, 0, 558, 598)

@spiral_config
def test_spiral_adjust_ratios(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 798, 598)
    manager.test_window('two')
    assert_dimensions(manager, 400, 0, 398, 598)
    manager.test_window('three')
    assert_dimensions(manager, 400, 300, 398, 298)
    manager.c.layout.increase_ratio()
    assert_dimensions(manager, 480, 360, 318, 238)
    manager.c.layout.increase_ratio()
    assert_dimensions(manager, 560, 420, 238, 178)
    for _ in range(4):
        manager.c.layout.decrease_ratio()
    assert_dimensions(manager, 240, 180, 558, 418)
    manager.c.layout.reset()
    assert_dimensions(manager, 400, 300, 398, 298)