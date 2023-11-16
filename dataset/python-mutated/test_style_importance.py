from textual.app import App, ComposeResult
from textual.color import Color
from textual.containers import Container
from textual.css.scalar import ScalarOffset

class StyleApp(App[None]):
    CSS = '\n    Container {\n        border: round green !important;\n        outline: round green !important;\n        align: right bottom !important;\n        content-align: right bottom !important;\n        offset: 17 23 !important;\n        overflow: hidden hidden !important;\n        padding: 10 20 30 40 !important;\n        scrollbar-size: 23 42 !important;\n    }\n\n    Container.more-specific {\n        border: solid red;\n        outline: solid red;\n        align: center middle;\n        content-align: center middle;\n        offset: 0 0;\n        overflow: scroll scroll;\n        padding: 1 2 3 4;\n        scrollbar-size: 1 2;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Container(classes='more-specific')

async def test_border_importance():
    """Border without sides should support !important"""
    async with StyleApp().run_test() as pilot:
        border = pilot.app.query_one(Container).styles.border
        desired = ('round', Color.parse('green'))
        assert border.top == desired
        assert border.left == desired
        assert border.bottom == desired
        assert border.right == desired

async def test_outline_importance():
    """Outline without sides should support !important"""
    async with StyleApp().run_test() as pilot:
        outline = pilot.app.query_one(Container).styles.outline
        desired = ('round', Color.parse('green'))
        assert outline.top == desired
        assert outline.left == desired
        assert outline.bottom == desired
        assert outline.right == desired

async def test_align_importance():
    """Align without direction should support !important"""
    async with StyleApp().run_test() as pilot:
        assert pilot.app.query_one(Container).styles.align == ('right', 'bottom')

async def test_content_align_importance():
    """Content align without direction should support !important"""
    async with StyleApp().run_test() as pilot:
        assert pilot.app.query_one(Container).styles.content_align == ('right', 'bottom')

async def test_offset_importance():
    """Offset without direction should support !important"""
    async with StyleApp().run_test() as pilot:
        assert pilot.app.query_one(Container).styles.offset == ScalarOffset.from_offset((17, 23))

async def test_overflow_importance():
    """Overflow without direction should support !important"""
    async with StyleApp().run_test() as pilot:
        assert pilot.app.query_one(Container).styles.overflow_x == 'hidden'
        assert pilot.app.query_one(Container).styles.overflow_y == 'hidden'

async def test_padding_importance():
    """Padding without sides should support !important"""
    async with StyleApp().run_test() as pilot:
        padding = pilot.app.query_one(Container).styles.padding
        assert padding.top == 10
        assert padding.left == 40
        assert padding.bottom == 30
        assert padding.right == 20

async def test_scrollbar_size_importance():
    """Scrollbar size without direction should support !important"""
    async with StyleApp().run_test() as pilot:
        assert pilot.app.query_one(Container).styles.scrollbar_size_horizontal == 23
        assert pilot.app.query_one(Container).styles.scrollbar_size_vertical == 42