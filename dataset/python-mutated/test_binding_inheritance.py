"""Tests relating to key binding inheritance.

In here you'll find some tests for general key binding inheritance, but
there is an emphasis on the inheriting of movement key bindings as they (as
of the time of writing) hold a special place in the Widget hierarchy of
Textual.

<URL:https://github.com/Textualize/textual/issues/1343> holds much of the
background relating to this.
"""
from __future__ import annotations
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Static
MOVEMENT_KEYS = ['up', 'down', 'left', 'right', 'home', 'end', 'pageup', 'pagedown']

class NoBindings(App[None]):
    """An app with zero bindings."""

async def test_just_app_no_bindings() -> None:
    """An app with no bindings should have no bindings, other than ctrl+c."""
    async with NoBindings().run_test() as pilot:
        assert list(pilot.app._bindings.keys.keys()) == ['ctrl+c', 'ctrl+backslash']
        assert pilot.app._bindings.get_key('ctrl+c').priority is True

class AlphaBinding(App[None]):
    """An app with a simple alpha key binding."""
    BINDINGS = [Binding('a', 'a', 'a', priority=True)]

async def test_just_app_alpha_binding() -> None:
    """An app with a single binding should have just the one binding."""
    async with AlphaBinding().run_test() as pilot:
        assert sorted(pilot.app._bindings.keys.keys()) == sorted(['ctrl+c', 'ctrl+backslash', 'a'])
        assert pilot.app._bindings.get_key('ctrl+c').priority is True
        assert pilot.app._bindings.get_key('a').priority is True

class LowAlphaBinding(App[None]):
    """An app with a simple low-priority alpha key binding."""
    BINDINGS = [Binding('a', 'a', 'a', priority=False)]

async def test_just_app_low_priority_alpha_binding() -> None:
    """An app with a single low-priority binding should have just the one binding."""
    async with LowAlphaBinding().run_test() as pilot:
        assert sorted(pilot.app._bindings.keys.keys()) == sorted(['ctrl+c', 'ctrl+backslash', 'a'])
        assert pilot.app._bindings.get_key('ctrl+c').priority is True
        assert pilot.app._bindings.get_key('a').priority is False

class ScreenWithBindings(Screen):
    """A screen with a simple alpha key binding."""
    BINDINGS = [Binding('a', 'a', 'a', priority=True)]

class AppWithScreenThatHasABinding(App[None]):
    """An app with no extra bindings but with a custom screen with a binding."""
    SCREENS = {'main': ScreenWithBindings}

    def on_mount(self) -> None:
        if False:
            return 10
        self.push_screen('main')

async def test_app_screen_with_bindings() -> None:
    """Test a screen with a single key binding defined."""
    async with AppWithScreenThatHasABinding().run_test() as pilot:
        assert pilot.app.screen._bindings.get_key('a').priority is True

class ScreenWithLowBindings(Screen):
    """A screen with a simple low-priority alpha key binding."""
    BINDINGS = [Binding('a', 'a', 'a', priority=False)]

class AppWithScreenThatHasALowBinding(App[None]):
    """An app with no extra bindings but with a custom screen with a low-priority binding."""
    SCREENS = {'main': ScreenWithLowBindings}

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        self.push_screen('main')

async def test_app_screen_with_low_bindings() -> None:
    """Test a screen with a single low-priority key binding defined."""
    async with AppWithScreenThatHasALowBinding().run_test() as pilot:
        assert pilot.app.screen._bindings.get_key('a').priority is False

class AppKeyRecorder(App[None]):
    """Base application class that can be used to record keystrokes."""
    ALPHAS = 'abcxyz'
    'str: The alpha keys to test against.'
    ALL_KEYS = [*ALPHAS, *MOVEMENT_KEYS]
    'list[str]: All the test keys.'

    @staticmethod
    def make_bindings(action_prefix: str='') -> list[Binding]:
        if False:
            i = 10
            return i + 15
        'Make the binding list for testing an app.\n\n        Args:\n            action_prefix (str, optional): An optional prefix for the action name.\n\n        Returns:\n            list[Binding]: The resulting list of bindings.\n        '
        return [Binding(key, f"{action_prefix}record('{key}')", key) for key in [*AppKeyRecorder.ALPHAS, *MOVEMENT_KEYS]]

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialise the recording app.'
        super().__init__()
        self.pressed_keys: list[str] = []

    async def action_record(self, key: str) -> None:
        """Record a key, as used from a binding.

        Args:
            key (str): The name of the key to record.
        """
        self.pressed_keys.append(key)

    def all_recorded(self, marker_prefix: str='') -> None:
        if False:
            print('Hello World!')
        'Were all the bindings recorded from the presses?\n\n        Args:\n            marker_prefix (str, optional): An optional prefix for the result markers.\n        '
        assert self.pressed_keys == [f'{marker_prefix}{key}' for key in self.ALL_KEYS]

class AppWithMovementKeysBound(AppKeyRecorder):
    """An application with bindings."""
    BINDINGS = AppKeyRecorder.make_bindings()

async def test_pressing_alpha_on_app() -> None:
    """Test that pressing the alpha key, when it's bound on the app, results in an action fire."""
    async with AppWithMovementKeysBound().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALPHAS)
        await pilot.pause()
        assert pilot.app.pressed_keys == [*AppKeyRecorder.ALPHAS]

async def test_pressing_movement_keys_app() -> None:
    """Test that pressing the movement keys, when they're bound on the app, results in an action fire."""
    async with AppWithMovementKeysBound().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        await pilot.pause()
        pilot.app.all_recorded()

class FocusableWidgetWithBindings(Static, can_focus=True):
    """A widget that has its own bindings for the movement keys."""
    BINDINGS = AppKeyRecorder.make_bindings('local_')

    async def action_local_record(self, key: str) -> None:
        await self.app.action_record(f'locally_{key}')

class AppWithWidgetWithBindings(AppKeyRecorder):
    """A test app that composes with a widget that has movement bindings."""

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield FocusableWidgetWithBindings()

    def on_mount(self) -> None:
        if False:
            return 10
        self.query_one(FocusableWidgetWithBindings).focus()

async def test_focused_child_widget_with_movement_bindings() -> None:
    """A focused child widget with movement bindings should handle its own actions."""
    async with AppWithWidgetWithBindings().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('locally_')

class FocusableWidgetWithNoBindings(Static, can_focus=True):
    """A widget that can receive focus but has no bindings."""

class ScreenWithMovementBindings(Screen):
    """A screen that binds keys, including movement keys."""
    BINDINGS = AppKeyRecorder.make_bindings('screen_')

    async def action_screen_record(self, key: str) -> None:
        await self.app.action_record(f'screenly_{key}')

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield FocusableWidgetWithNoBindings()

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        self.query_one(FocusableWidgetWithNoBindings).focus()

class AppWithScreenWithBindingsWidgetNoBindings(AppKeyRecorder):
    """An app with a non-default screen that handles movement key bindings."""
    SCREENS = {'main': ScreenWithMovementBindings}

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.push_screen('main')

async def test_focused_child_widget_with_movement_bindings_on_screen() -> None:
    """A focused child widget, with movement bindings in the screen, should trigger screen actions."""
    async with AppWithScreenWithBindingsWidgetNoBindings().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('screenly_')

class ScreenWithMovementBindingsAndContainerAroundWidget(Screen):
    """A screen that binds keys, including movement keys."""
    BINDINGS = AppKeyRecorder.make_bindings('screen_')

    async def action_screen_record(self, key: str) -> None:
        await self.app.action_record(f'screenly_{key}')

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Container(FocusableWidgetWithNoBindings())

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        self.query_one(FocusableWidgetWithNoBindings).focus()

class AppWithScreenWithBindingsWrappedWidgetNoBindings(AppKeyRecorder):
    """An app with a non-default screen that handles movement key bindings."""
    SCREENS = {'main': ScreenWithMovementBindings}

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.push_screen('main')

async def test_contained_focused_child_widget_with_movement_bindings_on_screen() -> None:
    """A contained focused child widget, with movement bindings in the screen, should trigger screen actions."""
    async with AppWithScreenWithBindingsWrappedWidgetNoBindings().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('screenly_')

class WidgetWithBindingsNoInherit(Static, can_focus=True, inherit_bindings=False):
    """A widget that has its own bindings for the movement keys, no binding inheritance."""
    BINDINGS = AppKeyRecorder.make_bindings('local_')

    async def action_local_record(self, key: str) -> None:
        await self.app.action_record(f'locally_{key}')

class AppWithWidgetWithBindingsNoInherit(AppKeyRecorder):
    """A test app that composes with a widget that has movement bindings without binding inheritance."""

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield WidgetWithBindingsNoInherit()

    def on_mount(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(WidgetWithBindingsNoInherit).focus()

async def test_focused_child_widget_with_movement_bindings_no_inherit() -> None:
    """A focused child widget with movement bindings and inherit_bindings=False should handle its own actions."""
    async with AppWithWidgetWithBindingsNoInherit().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('locally_')

class FocusableWidgetWithNoBindingsNoInherit(Static, can_focus=True, inherit_bindings=False):
    """A widget that can receive focus but has no bindings and doesn't inherit bindings."""

class ScreenWithMovementBindingsNoInheritChild(Screen):
    """A screen that binds keys, including movement keys."""
    BINDINGS = AppKeyRecorder.make_bindings('screen_')

    async def action_screen_record(self, key: str) -> None:
        await self.app.action_record(f'screenly_{key}')

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield FocusableWidgetWithNoBindingsNoInherit()

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(FocusableWidgetWithNoBindingsNoInherit).focus()

class AppWithScreenWithBindingsWidgetNoBindingsNoInherit(AppKeyRecorder):
    """An app with a non-default screen that handles movement key bindings, child no-inherit."""
    SCREENS = {'main': ScreenWithMovementBindingsNoInheritChild}

    def on_mount(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.push_screen('main')

async def test_focused_child_widget_no_inherit_with_movement_bindings_on_screen() -> None:
    """A focused child widget, that doesn't inherit bindings, with movement bindings in the screen, should trigger screen actions."""
    async with AppWithScreenWithBindingsWidgetNoBindingsNoInherit().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('screenly_')

class FocusableWidgetWithEmptyBindingsNoInherit(Static, can_focus=True, inherit_bindings=False):
    """A widget that can receive focus but has empty bindings and doesn't inherit bindings."""
    BINDINGS = []

class ScreenWithMovementBindingsNoInheritEmptyChild(Screen):
    """A screen that binds keys, including movement keys."""
    BINDINGS = AppKeyRecorder.make_bindings('screen_')

    async def action_screen_record(self, key: str) -> None:
        await self.app.action_record(f'screenly_{key}')

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield FocusableWidgetWithEmptyBindingsNoInherit()

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(FocusableWidgetWithEmptyBindingsNoInherit).focus()

class AppWithScreenWithBindingsWidgetEmptyBindingsNoInherit(AppKeyRecorder):
    """An app with a non-default screen that handles movement key bindings, child no-inherit."""
    SCREENS = {'main': ScreenWithMovementBindingsNoInheritEmptyChild}

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        self.push_screen('main')

async def test_focused_child_widget_no_inherit_empty_bindings_with_movement_bindings_on_screen() -> None:
    """A focused child widget, that doesn't inherit bindings and sets BINDINGS empty, with movement bindings in the screen, should trigger screen actions."""
    async with AppWithScreenWithBindingsWidgetEmptyBindingsNoInherit().run_test() as pilot:
        await pilot.press(*AppKeyRecorder.ALL_KEYS)
        pilot.app.all_recorded('screenly_')

class PriorityOverlapWidget(Static, can_focus=True):
    """A focusable widget with a priority binding."""
    BINDINGS = [Binding('0', "app.record('widget_0')", '0', priority=False), Binding('a', "app.record('widget_a')", 'a', priority=False), Binding('b', "app.record('widget_b')", 'b', priority=False), Binding('c', "app.record('widget_c')", 'c', priority=True), Binding('d', "app.record('widget_d')", 'd', priority=False), Binding('e', "app.record('widget_e')", 'e', priority=True), Binding('f', "app.record('widget_f')", 'f', priority=True)]

class PriorityOverlapScreen(Screen):
    """A screen with a priority binding."""
    BINDINGS = [Binding('0', "app.record('screen_0')", '0', priority=False), Binding('a', "app.record('screen_a')", 'a', priority=False), Binding('b', "app.record('screen_b')", 'b', priority=True), Binding('c', "app.record('screen_c')", 'c', priority=False), Binding('d', "app.record('screen_d')", 'c', priority=True), Binding('e', "app.record('screen_e')", 'e', priority=False), Binding('f', "app.record('screen_f')", 'f', priority=True)]

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield PriorityOverlapWidget()

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(PriorityOverlapWidget).focus()

class PriorityOverlapApp(AppKeyRecorder):
    """An application with a priority binding."""
    BINDINGS = [Binding('0', "record('app_0')", '0', priority=False), Binding('a', "record('app_a')", 'a', priority=True), Binding('b', "record('app_b')", 'b', priority=False), Binding('c', "record('app_c')", 'c', priority=False), Binding('d', "record('app_d')", 'c', priority=True), Binding('e', "record('app_e')", 'e', priority=True), Binding('f', "record('app_f')", 'f', priority=False)]
    SCREENS = {'main': PriorityOverlapScreen}

    def on_mount(self) -> None:
        if False:
            return 10
        self.push_screen('main')

async def test_overlapping_priority_bindings() -> None:
    """Test an app stack with overlapping bindings."""
    async with PriorityOverlapApp().run_test() as pilot:
        await pilot.press(*'0abcdef')
        assert pilot.app.pressed_keys == ['widget_0', 'app_a', 'screen_b', 'widget_c', 'app_d', 'app_e', 'screen_f']

async def test_skip_action() -> None:
    """Test that a binding may be skipped by an action raising SkipAction"""

    class Handle(Widget, can_focus=True):
        BINDINGS = [('t', "test('foo')", 'Test')]

        def action_test(self, text: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.app.exit(text)
    no_handle_invoked = False

    class NoHandle(Widget, can_focus=True):
        BINDINGS = [('t', "test('bar')", 'Test')]

        def action_test(self, text: str) -> bool:
            if False:
                i = 10
                return i + 15
            nonlocal no_handle_invoked
            no_handle_invoked = True
            raise SkipAction()

    class SkipApp(App):

        def compose(self) -> ComposeResult:
            if False:
                for i in range(10):
                    print('nop')
            yield Handle(NoHandle())

        def on_mount(self) -> None:
            if False:
                while True:
                    i = 10
            self.query_one(NoHandle).focus()
    async with SkipApp().run_test() as pilot:
        assert pilot.app.query_one(NoHandle).has_focus
        await pilot.press('t')
        assert no_handle_invoked
        assert pilot.app.return_value == 'foo'