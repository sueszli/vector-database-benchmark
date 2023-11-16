import pytest
import libqtile.config
from libqtile import layout
from libqtile.confreader import Config
from test.layouts.layout_utils import assert_focus_path, assert_focused

class TreeTabConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a'), libqtile.config.Group('b'), libqtile.config.Group('c'), libqtile.config.Group('d')]
    layouts = [layout.TreeTab(sections=['Foo', 'Bar'])]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    screens = []
    follow_mouse_focus = False
treetab_config = pytest.mark.parametrize('manager', [TreeTabConfig], indirect=True)

@treetab_config
def test_window(manager):
    if False:
        for i in range(10):
            print('nop')
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('float1', floating=True)
    manager.test_window('float2', floating=True)
    manager.test_window('three')
    assert manager.c.layout.info()['clients'] == ['one', 'three', 'two']
    assert manager.c.layout.info()['sections'] == ['Foo', 'Bar']
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['three']], 'Bar': []}
    assert_focused(manager, 'three')
    manager.c.layout.up()
    assert_focused(manager, 'two')
    manager.c.layout.down()
    assert_focused(manager, 'three')
    manager.c.layout.move_up()
    assert manager.c.layout.info()['clients'] == ['one', 'three', 'two']
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['three'], ['two']], 'Bar': []}
    manager.c.layout.move_down()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['three']], 'Bar': []}
    manager.c.layout.up()
    manager.c.layout.section_down()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['three']], 'Bar': [['two']]}
    manager.c.layout.section_up()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['three'], ['two']], 'Bar': []}
    manager.c.layout.up()
    manager.c.layout.section_down()
    manager.c.layout.del_section('Bar')
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['three']]}
    manager.c.layout.add_section('Baz')
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['three']], 'Baz': []}
    manager.c.layout.del_section('Baz')
    manager.c.layout.move_left()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['three']]}
    manager.c.layout.move_right()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two', ['three']]]}
    manager.c.layout.move_right()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two', ['three']]]}
    manager.test_window('four')
    manager.c.layout.move_right()
    manager.c.layout.up()
    manager.test_window('five')
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two', ['three', ['four']], ['five']]]}
    manager.c.layout.up()
    manager.c.layout.up()
    manager.c.layout.collapse_branch()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two', ['three'], ['five']]]}
    assert_focus_path(manager, 'five', 'float1', 'float2', 'one', 'two', 'three')
    manager.c.layout.expand_branch()
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two', ['three', ['four']], ['five']]]}
    assert_focus_path(manager, 'four', 'five', 'float1', 'float2', 'one', 'two', 'three')

@treetab_config
def test_sort_windows(manager):
    if False:
        i = 10
        return i + 15
    manager.test_window('one')
    manager.test_window('two')
    manager.test_window('101')
    manager.test_window('102')
    manager.test_window('103')
    assert manager.c.layout.info()['client_trees'] == {'Foo': [['one'], ['two'], ['101'], ['102'], ['103']], 'Bar': []}
    "\n    # TODO how to serialize a function object? i.e. `sorter`:\n\n    def sorter(window):\n        try:\n            if int(window.name) % 2 == 0:\n                return 'Even'\n            else:\n                return 'Odd'\n        except ValueError:\n            return 'Bar'\n\n    manager.c.layout.sort_windows(sorter)\n    assert manager.c.layout.info()['client_trees'] == {\n        'Foo': [],\n        'Bar': [['one'], ['two']],\n        'Even': [['102']],\n        'Odd': [['101'], ['103']]\n    }\n    "