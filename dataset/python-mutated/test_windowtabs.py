import pytest
import libqtile.config
from libqtile import bar, layout, widget
from libqtile.config import Screen
from libqtile.confreader import Config

class WindowTabsConfig(Config):
    auto_fullscreen = True
    groups = [libqtile.config.Group('a'), libqtile.config.Group('b')]
    layouts = [layout.Stack()]
    floating_layout = libqtile.resources.default_config.floating_layout
    keys = []
    mouse = []
    fake_screens = [Screen(top=bar.Bar([widget.WindowTabs()], 24), bottom=bar.Bar([widget.WindowTabs(selected='!!')], 24), x=0, y=0, width=900, height=960)]
    screens = []
windowtabs_config = pytest.mark.parametrize('manager', [WindowTabsConfig], indirect=True)

@windowtabs_config
def test_single_window_states(manager):
    if False:
        i = 10
        return i + 15

    def widget_text():
        if False:
            for i in range(10):
                print('nop')
        return manager.c.bar['top'].info()['widgets'][0]['text']
    assert widget_text() == ''
    proc = manager.test_window('one')
    assert widget_text() == '<b>one</b>'
    manager.c.window.toggle_maximize()
    assert widget_text() == '<b>[] one</b>'
    manager.c.window.toggle_minimize()
    assert widget_text() == '<b>_ one</b>'
    manager.c.window.toggle_minimize()
    manager.c.window.toggle_floating()
    assert widget_text() == '<b>V one</b>'
    manager.kill_window(proc)
    assert widget_text() == ''

@windowtabs_config
def test_multiple_windows(manager):
    if False:
        while True:
            i = 10

    def widget_text():
        if False:
            return 10
        return manager.c.bar['top'].info()['widgets'][0]['text']
    window_one = manager.test_window('one')
    assert widget_text() == '<b>one</b>'
    window_two = manager.test_window('two')
    assert widget_text() in ['<b>two</b> | one', 'one | <b>two</b>']
    manager.c.layout.next()
    assert widget_text() in ['<b>one</b> | two', 'two | <b>one</b>']
    manager.kill_window(window_one)
    assert widget_text() == '<b>two</b>'
    manager.kill_window(window_two)
    assert widget_text() == ''

@windowtabs_config
def test_selected(manager):
    if False:
        i = 10
        return i + 15

    def widget_text():
        if False:
            for i in range(10):
                print('nop')
        return manager.c.bar['bottom'].info()['widgets'][0]['text']
    window_one = manager.test_window('one')
    assert widget_text() == '!!one!!'
    manager.kill_window(window_one)
    assert widget_text() == ''

@windowtabs_config
def test_escaping_text(manager):
    if False:
        print('Hello World!')
    '\n    Ampersands can cause a crash if not escaped before passing to\n    pangocffi.parse_markup.\n    Test that the widget can parse text safely.\n    '
    manager.test_window('Text & Text')
    assert manager.c.widget['windowtabs'].info()['text'] == '<b>Text &amp; Text</b>'