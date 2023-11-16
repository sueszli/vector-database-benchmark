from string import punctuation
import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Center, Middle
from textual.pilot import OutOfBounds
from textual.widgets import Button, Label
KEY_CHARACTERS_TO_TEST = 'akTW03' + punctuation
'Test some "simple" characters (letters + digits) and all punctuation.'

class CenteredButtonApp(App):
    CSS = '  # Ensure the button is 16 x 3\n    Button {\n        min-width: 16;\n        max-width: 16;\n        width: 16;\n        min-height: 3;\n        max-height: 3;\n        height: 3;\n    }\n    '

    def compose(self):
        if False:
            for i in range(10):
                print('nop')
        with Center():
            with Middle():
                yield Button()

class ManyLabelsApp(App):
    """Auxiliary app with a button following many labels."""
    AUTO_FOCUS = None

    def compose(self):
        if False:
            while True:
                i = 10
        for idx in range(100):
            yield Label(f'label {idx}', id=f'label{idx}')
        yield Button()

async def test_pilot_press_ascii_chars():
    """Test that the pilot can press most ASCII characters as keys."""
    keys_pressed = []

    class TestApp(App[None]):

        def on_key(self, event: events.Key) -> None:
            if False:
                for i in range(10):
                    print('nop')
            keys_pressed.append(event.character)
    async with TestApp().run_test() as pilot:
        for char in KEY_CHARACTERS_TO_TEST:
            await pilot.press(char)
            assert keys_pressed[-1] == char

async def test_pilot_exception_catching_compose():
    """Ensuring that test frameworks are aware of exceptions
    inside compose methods when running via Pilot run_test()."""

    class FailingApp(App):

        def compose(self) -> ComposeResult:
            if False:
                i = 10
                return i + 15
            1 / 0
            yield Label('Beep')
    with pytest.raises(ZeroDivisionError):
        async with FailingApp().run_test():
            pass

async def test_pilot_exception_catching_action():
    """Ensure that exceptions inside action handlers are presented
    to the test framework when running via Pilot run_test()."""

    class FailingApp(App):
        BINDINGS = [Binding('b', 'beep', 'beep')]

        def action_beep(self) -> None:
            if False:
                return 10
            1 / 0
    with pytest.raises(ZeroDivisionError):
        async with FailingApp().run_test() as pilot:
            await pilot.press('b')

async def test_pilot_click_screen():
    """Regression test for https://github.com/Textualize/textual/issues/3395.

    Check we can use `Screen` as a selector for a click."""
    async with App().run_test() as pilot:
        await pilot.click('Screen')

async def test_pilot_hover_screen():
    """Regression test for https://github.com/Textualize/textual/issues/3395.

    Check we can use `Screen` as a selector for a hover."""
    async with App().run_test() as pilot:
        await pilot.hover('Screen')

@pytest.mark.parametrize(['method', 'screen_size', 'offset'], [('click', (80, 24), (100, 12)), ('click', (80, 24), (100, 36)), ('click', (80, 24), (50, 36)), ('click', (80, 24), (-10, 36)), ('click', (80, 24), (-10, 12)), ('click', (80, 24), (-10, -2)), ('click', (80, 24), (50, -2)), ('click', (80, 24), (100, -2)), ('click', (5, 5), (7, 3)), ('click', (5, 5), (7, 7)), ('click', (5, 5), (3, 7)), ('click', (5, 5), (-1, 7)), ('click', (5, 5), (-1, 3)), ('click', (5, 5), (-1, -1)), ('click', (5, 5), (3, -1)), ('click', (5, 5), (7, -1)), ('hover', (80, 24), (100, 12)), ('hover', (80, 24), (100, 36)), ('hover', (80, 24), (50, 36)), ('hover', (80, 24), (-10, 36)), ('hover', (80, 24), (-10, 12)), ('hover', (80, 24), (-10, -2)), ('hover', (80, 24), (50, -2)), ('hover', (80, 24), (100, -2)), ('hover', (5, 5), (7, 3)), ('hover', (5, 5), (7, 7)), ('hover', (5, 5), (3, 7)), ('hover', (5, 5), (-1, 7)), ('hover', (5, 5), (-1, 3)), ('hover', (5, 5), (-1, -1)), ('hover', (5, 5), (3, -1)), ('hover', (5, 5), (7, -1))])
async def test_pilot_target_outside_screen_errors(method, screen_size, offset):
    """Make sure that targeting a click/hover completely outside of the screen errors."""
    app = App()
    async with app.run_test(size=screen_size) as pilot:
        pilot_method = getattr(pilot, method)
        with pytest.raises(OutOfBounds):
            await pilot_method(offset=offset)

@pytest.mark.parametrize(['method', 'offset'], [('click', (0, 0)), ('click', (40, 0)), ('click', (79, 0)), ('click', (79, 12)), ('click', (79, 23)), ('click', (40, 23)), ('click', (40, 23)), ('click', (0, 12)), ('click', (40, 12)), ('hover', (0, 0)), ('hover', (40, 0)), ('hover', (79, 0)), ('hover', (79, 12)), ('hover', (79, 23)), ('hover', (40, 23)), ('hover', (40, 23)), ('hover', (0, 12)), ('hover', (40, 12))])
async def test_pilot_target_inside_screen_is_fine_with_correct_coordinate_system(method, offset):
    """Make sure that the coordinate system for the click is the correct one.

    Especially relevant because I kept getting confused about the way it works.
    """
    app = ManyLabelsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.query_one('#label99').scroll_visible(animate=False)
        await pilot.pause()
        pilot_method = getattr(pilot, method)
        await pilot_method(offset=offset)

@pytest.mark.parametrize(['method', 'target'], [('click', '#label0'), ('click', '#label90'), ('click', Button), ('hover', '#label0'), ('hover', '#label90'), ('hover', Button)])
async def test_pilot_target_on_widget_that_is_not_visible_errors(method, target):
    """Make sure that clicking a widget that is not scrolled into view raises an error."""
    app = ManyLabelsApp()
    async with app.run_test(size=(80, 5)) as pilot:
        app.query_one('#label50').scroll_visible(animate=False)
        await pilot.pause()
        pilot_method = getattr(pilot, method)
        with pytest.raises(OutOfBounds):
            await pilot_method(target)

@pytest.mark.parametrize('method', ['click', 'hover'])
async def test_pilot_target_widget_under_another_widget(method):
    """The targeting method should return False when the targeted widget is covered."""

    class ObscuredButton(App):
        CSS = '\n        Label {\n            width: 30;\n            height: 5;\n        }\n        '

        def compose(self):
            if False:
                i = 10
                return i + 15
            yield Button()
            yield Label()

        def on_mount(self):
            if False:
                print('Hello World!')
            self.query_one(Label).styles.offset = (0, -3)
    app = ObscuredButton()
    async with app.run_test() as pilot:
        await pilot.pause()
        pilot_method = getattr(pilot, method)
        assert await pilot_method(Button) is False

@pytest.mark.parametrize('method', ['click', 'hover'])
async def test_pilot_target_visible_widget(method):
    """The targeting method should return True when the targeted widget is hit."""

    class ObscuredButton(App):

        def compose(self):
            if False:
                while True:
                    i = 10
            yield Button()
    app = ObscuredButton()
    async with app.run_test() as pilot:
        await pilot.pause()
        pilot_method = getattr(pilot, method)
        assert await pilot_method(Button) is True

@pytest.mark.parametrize(['method', 'offset'], [('click', (0, 0)), ('click', (2, 0)), ('click', (10, 23)), ('click', (70, 0)), ('hover', (0, 0)), ('hover', (2, 0)), ('hover', (10, 23)), ('hover', (70, 0))])
async def test_pilot_target_screen_always_true(method, offset):
    app = ManyLabelsApp()
    async with app.run_test(size=(80, 24)) as pilot:
        pilot_method = getattr(pilot, method)
        assert await pilot_method(offset=offset) is True