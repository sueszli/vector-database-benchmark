from textual.app import App, ComposeResult
from textual.color import Color
from textual.widget import Widget

class Base(Widget):
    DEFAULT_CSS = '\n    Base {\n        color: magenta;\n    }\n    '

class CustomWidget1(Base):
    DEFAULT_CSS = '\n    CustomWidget1 {\n        background: red\n    }\n    '

class CustomWidget2(CustomWidget1):
    DEFAULT_CSS = '\n    CustomWidget2 {\n        background: initial;\n    }\n    '

class CustomWidget3(CustomWidget2):
    pass

async def test_initial_default():

    class InitialApp(App):

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            yield Base(id='base')
            yield CustomWidget1(id='custom1')
            yield CustomWidget2(id='custom2')
    app = InitialApp()
    async with app.run_test():
        base = app.query_one('#base', Base)
        custom1 = app.query_one('#custom1', CustomWidget1)
        custom2 = app.query_one('#custom2', CustomWidget2)
        default_background = base.styles.background
        assert default_background == Color.parse('rgba(0,0,0,0)')
        assert custom1.styles.background == Color.parse('red')
        assert custom2.styles.background == default_background

async def test_initial():

    class InitialApp(App):
        CSS = '\n        CustomWidget1 {\n            color: red;\n        }\n\n        CustomWidget2 {\n           color: initial;\n        }\n\n        CustomWidget3 {\n            color: blue;\n        }\n        '

        def compose(self) -> ComposeResult:
            if False:
                print('Hello World!')
            yield Base(id='base')
            yield CustomWidget1(id='custom1')
            yield CustomWidget2(id='custom2')
            yield CustomWidget3(id='custom3')
    app = InitialApp()
    async with app.run_test():
        base = app.query_one('#base')
        custom1 = app.query_one('#custom1')
        custom2 = app.query_one('#custom2')
        custom3 = app.query_one('#custom3')
        assert base.styles.color == Color.parse('magenta')
        assert custom1.styles.color == Color.parse('red')
        assert custom2.styles.color == Color.parse('magenta')
        assert custom3.styles.color == Color.parse('blue')