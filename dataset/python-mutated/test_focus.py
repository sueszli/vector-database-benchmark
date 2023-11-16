import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button

class Focusable(Widget, can_focus=True):
    pass

class NonFocusable(Widget, can_focus=False, can_focus_children=False):
    pass

class ChildrenFocusableOnly(Widget, can_focus=False, can_focus_children=True):
    pass

@pytest.fixture
def screen() -> Screen:
    if False:
        return 10
    app = App()
    app._set_active()
    app.push_screen(Screen())
    screen = app.screen
    screen._add_children(Focusable(id='foo', classes='a'), NonFocusable(id='bar'), Focusable(Focusable(id='Paul', classes='c'), id='container1', classes='b'), NonFocusable(Focusable(id='Jessica', classes='a'), id='container2'), Focusable(id='baz', classes='b'), ChildrenFocusableOnly(Focusable(id='child', classes='c')))
    return screen

def test_focus_chain():
    if False:
        while True:
            i = 10
    app = App()
    app._set_active()
    app.push_screen(Screen())
    screen = app.screen
    assert not screen.focus_chain
    app.screen._add_children(Focusable(id='foo'), NonFocusable(id='bar'), Focusable(Focusable(id='Paul'), id='container1'), NonFocusable(Focusable(id='Jessica'), id='container2'), Focusable(id='baz'), ChildrenFocusableOnly(Focusable(id='child')))
    focus_chain = [widget.id for widget in screen.focus_chain]
    assert focus_chain == ['foo', 'container1', 'Paul', 'baz', 'child']

def test_focus_next_and_previous(screen: Screen):
    if False:
        return 10
    assert screen.focus_next().id == 'foo'
    assert screen.focus_next().id == 'container1'
    assert screen.focus_next().id == 'Paul'
    assert screen.focus_next().id == 'baz'
    assert screen.focus_next().id == 'child'
    assert screen.focus_previous().id == 'baz'
    assert screen.focus_previous().id == 'Paul'
    assert screen.focus_previous().id == 'container1'
    assert screen.focus_previous().id == 'foo'

def test_focus_next_wrap_around(screen: Screen):
    if False:
        for i in range(10):
            print('nop')
    'Ensure focusing the next widget wraps around the focus chain.'
    screen.set_focus(screen.query_one('#child'))
    assert screen.focused.id == 'child'
    assert screen.focus_next().id == 'foo'

def test_focus_previous_wrap_around(screen: Screen):
    if False:
        while True:
            i = 10
    'Ensure focusing the previous widget wraps around the focus chain.'
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused.id == 'foo'
    assert screen.focus_previous().id == 'child'

def test_wrap_around_selector(screen: Screen):
    if False:
        print('Hello World!')
    'Ensure moving focus in both directions wraps around the focus chain.'
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused.id == 'foo'
    assert screen.focus_previous('#Paul').id == 'Paul'
    assert screen.focus_next('#foo').id == 'foo'

def test_no_focus_empty_selector(screen: Screen):
    if False:
        i = 10
        return i + 15
    'Ensure focus is cleared when selector matches nothing.'
    assert screen.focus_next('#bananas') is None
    assert screen.focus_previous('#bananas') is None
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused is not None
    assert screen.focus_next('bananas') is None
    assert screen.focused is None
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused is not None
    assert screen.focus_previous('bananas') is None
    assert screen.focused is None

def test_focus_next_and_previous_with_type_selector(screen: Screen):
    if False:
        while True:
            i = 10
    'Move focus with a selector that matches the currently focused node.'
    screen.set_focus(screen.query_one('#Paul'))
    assert screen.focused.id == 'Paul'
    assert screen.focus_next(Focusable).id == 'baz'
    assert screen.focus_next(Focusable).id == 'child'
    assert screen.focus_previous(Focusable).id == 'baz'
    assert screen.focus_previous(Focusable).id == 'Paul'
    assert screen.focus_previous(Focusable).id == 'container1'
    assert screen.focus_previous(Focusable).id == 'foo'

def test_focus_next_and_previous_with_str_selector(screen: Screen):
    if False:
        i = 10
        return i + 15
    'Move focus with a selector that matches the currently focused node.'
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused.id == 'foo'
    assert screen.focus_next('.a').id == 'foo'
    assert screen.focus_next('.c').id == 'Paul'
    assert screen.focus_next('.c').id == 'child'
    assert screen.focus_previous('.c').id == 'Paul'
    assert screen.focus_previous('.a').id == 'foo'

def test_focus_next_and_previous_with_type_selector_without_self():
    if False:
        i = 10
        return i + 15
    'Test moving the focus with a selector that does not match the currently focused node.'
    app = App()
    app._set_active()
    app.push_screen(Screen())
    screen = app.screen
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import Button, Input, Switch
    screen._add_children(VerticalScroll(Horizontal(Input(id='w3'), Switch(id='w4'), Input(id='w5'), Button(id='w6'), Switch(id='w7'), id='w2'), Horizontal(Button(id='w9'), Switch(id='w10'), Button(id='w11'), Input(id='w12'), Input(id='w13'), id='w8'), id='w1'))
    screen.set_focus(screen.query_one('#w3'))
    assert screen.focused.id == 'w3'
    assert screen.focus_next(Button).id == 'w6'
    assert screen.focus_next(Switch).id == 'w7'
    assert screen.focus_next(Input).id == 'w12'
    assert screen.focus_previous(Button).id == 'w11'
    assert screen.focus_previous(Switch).id == 'w10'
    assert screen.focus_previous(Button).id == 'w9'
    assert screen.focus_previous(Input).id == 'w5'

def test_focus_next_and_previous_with_str_selector_without_self(screen: Screen):
    if False:
        print('Hello World!')
    'Test moving the focus with a selector that does not match the currently focused node.'
    screen.set_focus(screen.query_one('#foo'))
    assert screen.focused.id == 'foo'
    assert screen.focus_next('.c').id == 'Paul'
    assert screen.focus_next('.b').id == 'baz'
    assert screen.focus_next('.c').id == 'child'
    assert screen.focus_previous('.a').id == 'foo'
    assert screen.focus_previous('.a').id == 'foo'
    assert screen.focus_previous('.b').id == 'baz'

async def test_focus_does_not_move_to_invisible_widgets():
    """Make sure invisible widgets don't get focused by accident.

    This is kind of a regression test for https://github.com/Textualize/textual/issues/3053,
    but not really.
    """

    class MyApp(App):
        CSS = '#inv { visibility: hidden; }'

        def compose(self):
            if False:
                print('Hello World!')
            yield Button('one', id='one')
            yield Button('two', id='inv')
            yield Button('three', id='three')
    app = MyApp()
    async with app.run_test():
        assert app.focused.id == 'one'
        assert app.screen.focus_next().id == 'three'

async def test_focus_moves_to_visible_widgets_inside_invisible_containers():
    """Regression test for https://github.com/Textualize/textual/issues/3053."""

    class MyApp(App):
        CSS = '\n        #inv { visibility: hidden; }\n        #three { visibility: visible; }\n        '

        def compose(self):
            if False:
                return 10
            yield Button(id='one')
            with Container(id='inv'):
                yield Button(id='three')
    app = MyApp()
    async with app.run_test():
        assert app.focused.id == 'one'
        assert app.screen.focus_next().id == 'three'

async def test_focus_chain_handles_inherited_visibility():
    """Regression test for https://github.com/Textualize/textual/issues/3053

    This is more or less a test for the interactions between #3053 and #3071.
    We want to make sure that the focus chain is computed correctly when going through
    a DOM with containers with all sorts of visibilities set.
    """

    class W(Widget):
        can_focus = True
    w1 = W(id='one')
    c2 = Container(id='two')
    w3 = W(id='three')
    c4 = Container(id='four')
    w5 = W(id='five')
    c6 = Container(id='six')
    w7 = W(id='seven')
    c8 = Container(id='eight')
    w9 = W(id='nine')
    w10 = W(id='ten')
    w11 = W(id='eleven')
    w12 = W(id='twelve')
    w13 = W(id='thirteen')

    class InheritedVisibilityApp(App[None]):
        CSS = '\n        #four, #eight, #ten {\n            visibility: visible;\n        }\n\n        #six, #thirteen {\n            visibility: hidden;\n        }\n        '

        def compose(self):
            if False:
                return 10
            yield w1
            with c2:
                yield w3
                with c4:
                    yield w5
                    with c6:
                        yield w7
                        with c8:
                            yield w9
                        yield w10
                    yield w11
                yield w12
            yield w13
    app = InheritedVisibilityApp()
    async with app.run_test():
        focus_chain = app.screen.focus_chain
        assert focus_chain == [w1, w3, w5, w9, w10, w11, w12]

async def test_focus_pseudo_class():
    """Test focus and blue pseudo classes"""

    class FocusApp(App):
        AUTO_FOCUS = None

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            yield Button('Hello')
    app = FocusApp()
    async with app.run_test() as pilot:
        button = app.query_one(Button)
        classes = list(button.get_pseudo_classes())
        assert 'blur' in classes
        assert 'focus' not in classes
        button.focus()
        await pilot.pause()
        classes = list(button.get_pseudo_classes())
        assert 'blur' not in classes
        assert 'focus' in classes