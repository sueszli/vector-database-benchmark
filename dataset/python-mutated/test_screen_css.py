from textual.app import App
from textual.color import Color
from textual.screen import Screen
from textual.widgets import Label
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)

class BaseScreen(Screen):

    def compose(self):
        if False:
            return 10
        yield Label('Hello, world!', id='app-css')
        yield Label('Hello, world!', id='screen-css-path')
        yield Label('Hello, world!', id='screen-css')

class ScreenWithCSS(Screen):
    SCOPED_CSS = False
    CSS = '\n    #screen-css {\n        background: #ff0000;\n    }\n    '
    CSS_PATH = 'test_screen_css.tcss'

    def compose(self):
        if False:
            i = 10
            return i + 15
        yield Label('Hello, world!', id='app-css')
        yield Label('Hello, world!', id='screen-css-path')
        yield Label('Hello, world!', id='screen-css')

class BaseApp(App):
    """Base app for testing screen CSS when pushing screens."""
    CSS = '\n    #app-css {\n        background: #00ff00;\n    }\n    #screen-css-path {\n        background: #00ff00;\n    }\n    #screen-css {\n        background: #00ff00;\n    }\n    '

    def on_mount(self):
        if False:
            for i in range(10):
                print('nop')
        self.push_screen(BaseScreen())

class SwitchBaseApp(BaseApp):
    """Base app for testing screen CSS when switching a screen."""

    def on_mount(self):
        if False:
            return 10
        self.push_screen(BaseScreen())

def check_colors_before_screen_css(app: BaseApp):
    if False:
        while True:
            i = 10
    assert app.query_one('#app-css').styles.background == GREEN
    assert app.query_one('#screen-css-path').styles.background == GREEN
    assert app.query_one('#screen-css').styles.background == GREEN

def check_colors_after_screen_css(app: BaseApp):
    if False:
        return 10
    assert app.query_one('#app-css').styles.background == GREEN
    assert app.query_one('#screen-css-path').styles.background == BLUE
    assert app.query_one('#screen-css').styles.background == RED

async def test_screen_pushing_and_popping_does_not_reparse_css():
    """Check that pushing and popping the same screen doesn't trigger CSS reparses."""

    class MyApp(BaseApp):

        def key_p(self):
            if False:
                print('Hello World!')
            self.push_screen(ScreenWithCSS())

        def key_o(self):
            if False:
                for i in range(10):
                    print('nop')
            self.pop_screen()
    counter = 0

    def reparse_wrapper(reparse):
        if False:
            for i in range(10):
                print('nop')

        def _reparse(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal counter
            counter += 1
            return reparse(*args, **kwargs)
        return _reparse
    app = MyApp()
    app.stylesheet.reparse = reparse_wrapper(app.stylesheet.reparse)
    async with app.run_test() as pilot:
        await pilot.press('p')
        await pilot.press('o')
        await pilot.press('p')
        await pilot.press('o')
        await pilot.press('p')
        await pilot.press('o')
        await pilot.press('p')
        await pilot.press('o')
        assert counter == 1

async def test_screen_css_push_screen_instance():
    """Check that screen CSS is loaded and applied when pushing a screen instance."""

    class MyApp(BaseApp):

        def key_p(self):
            if False:
                while True:
                    i = 10
            self.push_screen(ScreenWithCSS())

        def key_o(self):
            if False:
                i = 10
                return i + 15
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_push_screen_instance_by_name():
    """Check that screen CSS is loaded and applied when pushing a screen name that points to a screen instance."""

    class MyApp(BaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS()}

        def key_p(self):
            if False:
                i = 10
                return i + 15
            self.push_screen('screenwithcss')

        def key_o(self):
            if False:
                i = 10
                return i + 15
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_push_screen_type_by_name():
    """Check that screen CSS is loaded and applied when pushing a screen name that points to a screen class."""

    class MyApp(BaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS}

        def key_p(self):
            if False:
                for i in range(10):
                    print('nop')
            self.push_screen('screenwithcss')

        def key_o(self):
            if False:
                print('Hello World!')
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_screen_instance():
    """Check that screen CSS is loaded and applied when switching to a screen instance."""

    class MyApp(SwitchBaseApp):

        def key_p(self):
            if False:
                print('Hello World!')
            self.switch_screen(ScreenWithCSS())

        def key_o(self):
            if False:
                print('Hello World!')
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_screen_instance_by_name():
    """Check that screen CSS is loaded and applied when switching a screen name that points to a screen instance."""

    class MyApp(SwitchBaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS()}

        def key_p(self):
            if False:
                return 10
            self.switch_screen('screenwithcss')

        def key_o(self):
            if False:
                return 10
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_screen_type_by_name():
    """Check that screen CSS is loaded and applied when switching a screen name that points to a screen class."""

    class MyApp(SwitchBaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS}

        def key_p(self):
            if False:
                while True:
                    i = 10
            self.switch_screen('screenwithcss')

        def key_o(self):
            if False:
                return 10
            self.pop_screen()
    app = MyApp()
    async with app.run_test() as pilot:
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_mode_screen_instance():
    """Check that screen CSS is loaded and applied when switching to a mode with a screen instance."""

    class MyApp(BaseApp):
        MODES = {'base': BaseScreen(), 'mode': ScreenWithCSS()}

        def key_p(self):
            if False:
                for i in range(10):
                    print('nop')
            self.switch_mode('mode')

        def key_o(self):
            if False:
                for i in range(10):
                    print('nop')
            self.switch_mode('base')
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press('o')
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_mode_screen_instance_by_name():
    """Check that screen CSS is loaded and applied when switching to a mode with a screen instance name."""

    class MyApp(BaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS()}
        MODES = {'base': BaseScreen(), 'mode': 'screenwithcss'}

        def key_p(self):
            if False:
                i = 10
                return i + 15
            self.switch_mode('mode')

        def key_o(self):
            if False:
                print('Hello World!')
            self.switch_mode('base')
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press('o')
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)

async def test_screen_css_switch_mode_screen_type_by_name():
    """Check that screen CSS is loaded and applied when switching to a mode with a screen type name."""

    class MyApp(BaseApp):
        SCREENS = {'screenwithcss': ScreenWithCSS}
        MODES = {'base': BaseScreen(), 'mode': 'screenwithcss'}

        def key_p(self):
            if False:
                while True:
                    i = 10
            self.switch_mode('mode')

        def key_o(self):
            if False:
                print('Hello World!')
            self.switch_mode('base')
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press('o')
        check_colors_before_screen_css(app)
        await pilot.press('p')
        check_colors_after_screen_css(app)
        await pilot.press('o')
        check_colors_after_screen_css(app)