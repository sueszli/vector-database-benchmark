import pytest
import libqtile.config
from libqtile import layout
from libqtile.confreader import Config
from test.layouts.layout_utils import assert_dimensions, assert_focus_path, assert_focused

class MonadTallConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadTall()]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadtall_config = pytest.mark.parametrize('manager', [MonadTallConfig], indirect=True)

class MonadTallNCPBeforeCurrentConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadTall(new_client_position='before_current')]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadtallncpbeforecurrent_config = pytest.mark.parametrize('manager', [MonadTallNCPBeforeCurrentConfig], indirect=True)

class MonadTallNCPAfterCurrentConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadTall(new_client_position='after_current')]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadtallncpaftercurrent_config = pytest.mark.parametrize('manager', [MonadTallNCPAfterCurrentConfig], indirect=True)

class MonadTallNewCLientPositionBottomConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadTall(new_client_position='bottom')]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False

class MonadTallMarginsConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadTall(margin=4)]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadtallmargins_config = pytest.mark.parametrize('manager', [MonadTallMarginsConfig], indirect=True)

class MonadWideConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadWide()]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadwide_config = pytest.mark.parametrize('manager', [MonadWideConfig], indirect=True)

class MonadWideNewClientPositionTopConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadWide(new_client_position='top')]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False

class MonadWideMarginsConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadWide(margin=4)]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False

@monadtall_config
def test_tall_add_clients(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two']
    assert_focused(manager, 'two')
    manager.test_window('three')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three']
    assert_focused(manager, 'three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'four', 'three']
    assert_focused(manager, 'four')

@monadtallncpbeforecurrent_config
def test_tall_add_clients_before_current(manager):
    if False:
        i = 10
        return i + 15
    'Test add client with new_client_position = before_current.'
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['two', 'one']
    manager.c.layout.next()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['four', 'two', 'one']
    assert_focused(manager, 'four')

@monadtallncpaftercurrent_config
def test_tall_add_clients_after_current(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'four', 'three']
    assert_focused(manager, 'four')

@pytest.mark.parametrize('manager', [MonadTallNewCLientPositionBottomConfig], indirect=True)
def test_tall_add_clients_at_bottom(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'four']

@monadwide_config
def test_wide_add_clients(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    manager.test_window('two')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two']
    assert_focused(manager, 'two')
    manager.test_window('three')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three']
    assert_focused(manager, 'three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'four', 'three']
    assert_focused(manager, 'four')

@pytest.mark.parametrize('manager', [MonadWideNewClientPositionTopConfig], indirect=True)
def test_wide_add_clients_new_client_postion_top(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == ['one']
    assert_focused(manager, 'two')
    manager.test_window('three')
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['two', 'one']
    assert_focused(manager, 'three')
    manager.c.layout.next()
    assert_focused(manager, 'two')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    assert_focused(manager, 'four')

@monadtallmargins_config
def test_tall_margins(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert_dimensions(manager, 4, 4, 788, 588)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 404, 4, 388, 588)
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 4, 4, 392, 588)

@pytest.mark.parametrize('manager', [MonadWideMarginsConfig], indirect=True)
def test_wide_margins(manager):
    if False:
        return 10
    manager.test_window('one')
    assert_dimensions(manager, 4, 4, 788, 588)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 4, 304, 788, 288)
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 4, 4, 788, 292)

@monadtall_config
def test_tall_growmain_solosecondary(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 396, 596)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 436, 596)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 396, 596)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 596, 596)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 196, 596)

@monadwide_config
def test_wide_growmain_solosecondary(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 796, 296)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 796, 326)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 796, 296)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 796, 446)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 796, 146)

@monadtall_config
def test_tall_growmain_multiplesecondary(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 396, 596)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 436, 596)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 396, 596)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 596, 596)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 196, 596)

@monadwide_config
def test_wide_growmain_multiplesecondary(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    manager.c.layout.previous()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 796, 296)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 796, 326)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 796, 296)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 0, 796, 446)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 0, 796, 146)

@monadtall_config
def test_tall_growsecondary_solosecondary(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 400, 0, 396, 596)
    manager.c.layout.grow()
    assert_dimensions(manager, 360, 0, 436, 596)
    manager.c.layout.shrink()
    assert_dimensions(manager, 400, 0, 396, 596)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 200, 0, 596, 596)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 600, 0, 196, 596)

@monadwide_config
def test_wide_growsecondary_solosecondary(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 300, 796, 296)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 270, 796, 326)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 300, 796, 296)
    for _ in range(10):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 150, 796, 446)
    for _ in range(10):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 450, 796, 146)

@monadtall_config
def test_tall_growsecondary_multiplesecondary(manager):
    if False:
        return 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 400, 0, 396, 296)
    manager.c.layout.grow()
    assert_dimensions(manager, 400, 0, 396, 316)
    manager.c.layout.shrink()
    assert_dimensions(manager, 400, 0, 396, 296)
    for _ in range(20):
        manager.c.layout.grow()
    assert_dimensions(manager, 400, 0, 396, 511)
    for _ in range(40):
        manager.c.layout.shrink()
    assert_dimensions(manager, 400, 0, 396, 85)

@monadwide_config
def test_wide_growsecondary_multiplesecondary(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.previous()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 300, 396, 296)
    manager.c.layout.grow()
    assert_dimensions(manager, 0, 300, 416, 296)
    manager.c.layout.shrink()
    assert_dimensions(manager, 0, 300, 396, 296)
    for _ in range(20):
        manager.c.layout.grow()
    assert_dimensions(manager, 0, 300, 710, 296)
    for _ in range(40):
        manager.c.layout.shrink()
    assert_dimensions(manager, 0, 300, 85, 296)

@monadtall_config
def test_tall_flip(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.next()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 396, 596)
    manager.c.layout.next()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 400, 0, 396, 296)
    manager.c.layout.next()
    assert_focused(manager, 'three')
    assert_dimensions(manager, 400, 300, 396, 296)
    manager.c.layout.flip()
    manager.c.layout.next()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 400, 0, 396, 596)
    manager.c.layout.next()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 0, 396, 296)
    manager.c.layout.next()
    assert_focused(manager, 'three')
    assert_dimensions(manager, 0, 300, 396, 296)

@monadwide_config
def test_wide_flip(manager):
    if False:
        return 10
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.c.layout.next()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 0, 796, 296)
    manager.c.layout.next()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 300, 396, 296)
    manager.c.layout.next()
    assert_focused(manager, 'three')
    assert_dimensions(manager, 400, 300, 396, 296)
    manager.c.layout.flip()
    manager.c.layout.next()
    assert_focused(manager, 'one')
    assert_dimensions(manager, 0, 300, 796, 296)
    manager.c.layout.next()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 0, 396, 296)
    manager.c.layout.next()
    assert_focused(manager, 'three')
    assert_dimensions(manager, 400, 0, 396, 296)

@monadtall_config
def test_tall_set_and_reset(manager):
    if False:
        return 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 400, 0, 396, 596)
    manager.c.layout.set_ratio(0.75)
    assert_focused(manager, 'two')
    assert_dimensions(manager, 600, 0, 196, 596)
    manager.c.layout.set_ratio(0.25)
    assert_focused(manager, 'two')
    assert_dimensions(manager, 200, 0, 596, 596)
    manager.c.layout.reset()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 400, 0, 396, 596)

@monadwide_config
def test_wide_set_and_reset(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    assert_dimensions(manager, 0, 0, 796, 596)
    manager.test_window('two')
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 300, 796, 296)
    manager.c.layout.set_ratio(0.75)
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 450, 796, 146)
    manager.c.layout.set_ratio(0.25)
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 150, 796, 446)
    manager.c.layout.reset()
    assert_focused(manager, 'two')
    assert_dimensions(manager, 0, 300, 796, 296)

@monadtall_config
def test_tall_shuffle(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'four']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'four', 'three']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['four', 'two', 'three']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == ['one', 'two', 'three']

@monadwide_config
def test_wide_shuffle(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'four']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'four', 'three']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['four', 'two', 'three']
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == ['one', 'two', 'three']

@monadtall_config
def test_tall_swap(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('focused')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'focused']
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'one']
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == ['focused', 'three', 'one']
    manager.c.layout.flip()
    manager.c.layout.shuffle_down()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == ['three', 'focused', 'one']
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['focused', 'two', 'one']
    manager.c.layout.swap_main()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_left()
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['focused', 'two', 'one']

@monadwide_config
def test_wide_swap(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('focused')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'focused']
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['two', 'three', 'one']
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == ['focused', 'three', 'one']
    manager.c.layout.flip()
    manager.c.layout.shuffle_down()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == ['three', 'focused', 'one']
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['focused', 'two', 'one']
    manager.c.layout.swap_main()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_left()
    assert manager.c.layout.info()['main'] == 'focused'
    assert manager.c.layout.info()['secondary'] == ['three', 'two', 'one']
    manager.c.layout.swap_right()
    manager.c.layout.swap_right()
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == ['focused', 'two', 'one']

@monadtall_config
def test_tall_window_focus_cycle(manager):
    if False:
        return 10
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('float1')
    manager.c.window.toggle_floating()
    manager.test_window('float2')
    manager.c.window.toggle_floating()
    manager.test_window('three')
    assert manager.c.layout.info()['clients'] == ['one', 'two', 'three']
    assert_focused(manager, 'three')
    assert_focus_path(manager, 'float1', 'float2', 'one', 'two', 'three')

@monadwide_config
def test_wide_window_focus_cycle(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('float1')
    manager.c.window.toggle_floating()
    manager.test_window('float2')
    manager.c.window.toggle_floating()
    manager.test_window('three')
    assert manager.c.layout.info()['clients'] == ['one', 'two', 'three']
    assert_focused(manager, 'three')
    assert_focus_path(manager, 'float1', 'float2', 'one', 'two', 'three')

class MonadThreeColConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a')]
    layouts = [layout.MonadThreeCol()]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
monadthreecol_config = pytest.mark.parametrize('manager', [MonadThreeColConfig], indirect=True)

@monadthreecol_config
def test_three_col_add_clients(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    assert manager.c.layout.info()['main'] == 'one'
    assert manager.c.layout.info()['secondary'] == dict(left=[], right=[])
    manager.test_window('two')
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == dict(left=['one'], right=[])
    assert_focused(manager, 'two')
    manager.test_window('three')
    assert manager.c.layout.info()['main'] == 'three'
    assert manager.c.layout.info()['secondary'] == dict(left=['two'], right=['one'])
    assert_focused(manager, 'three')
    manager.test_window('four')
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == dict(left=['three', 'two'], right=['one'])
    assert_focused(manager, 'four')
    manager.test_window('five')
    assert manager.c.layout.info()['main'] == 'five'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['two', 'one'])
    assert_focused(manager, 'five')
    manager.c.layout.next()
    assert_focused(manager, 'four')
    manager.c.layout.next()
    assert_focused(manager, 'three')
    manager.c.layout.next()
    assert_focused(manager, 'two')
    manager.c.layout.next()
    assert_focused(manager, 'one')

@monadthreecol_config
def test_three_col_shuffle(manager):
    if False:
        while True:
            i = 10
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('four')
    manager.test_window('five')
    manager.c.layout.shuffle_right()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['five', 'one'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_down()
    assert manager.c.layout.info()['main'] == 'two'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['one', 'five'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_left()
    assert manager.c.layout.info()['main'] == 'five'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['one', 'two'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_left()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == dict(left=['five', 'three'], right=['one', 'two'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_down()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == dict(left=['three', 'five'], right=['one', 'two'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_up()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == dict(left=['five', 'three'], right=['one', 'two'])
    assert_focused(manager, 'five')
    manager.c.layout.shuffle_right()
    assert manager.c.layout.info()['main'] == 'five'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['one', 'two'])
    assert_focused(manager, 'five')

@monadthreecol_config
def test_three_col_swap_main(manager):
    if False:
        print('Hello World!')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('three')
    manager.test_window('four')
    manager.test_window('five')
    manager.c.layout.next()
    manager.c.layout.swap_main()
    assert manager.c.layout.info()['main'] == 'four'
    assert manager.c.layout.info()['secondary'] == dict(left=['five', 'three'], right=['two', 'one'])
    assert_focused(manager, 'four')
    manager.c.layout.next()
    manager.c.layout.swap_main()
    assert manager.c.layout.info()['main'] == 'five'
    assert manager.c.layout.info()['secondary'] == dict(left=['four', 'three'], right=['two', 'one'])
    assert_focused(manager, 'five')