from textual.app import App, ComposeResult
from textual.widgets import Button, Static

async def test_text_style_inheritance():
    """Check that changes to text style are inherited in children."""

    class FocusableThing(Static, can_focus=True):
        DEFAULT_CSS = '\n        FocusableThing {\n            text-style: bold;\n        }\n\n        FocusableThing:focus {\n            text-style: bold reverse;\n        }\n        '

        def compose(self) -> ComposeResult:
            if False:
                return 10
            yield Static('test', id='child-of-focusable-thing')

    class InheritanceApp(App):

        def compose(self) -> ComposeResult:
            if False:
                for i in range(10):
                    print('nop')
            yield Button('button1')
            yield FocusableThing()
            yield Button('button2')
    app = InheritanceApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        child = app.query_one('#child-of-focusable-thing')
        assert child.rich_style.bold
        assert not child.rich_style.reverse
        await pilot.press('tab')
        await pilot.pause()
        assert child.rich_style.bold
        assert child.rich_style.reverse