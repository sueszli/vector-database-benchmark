from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import Input

class InputApp(App):
    CSS = 'Input { padding: 4 8 }'

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Input('こんにちは!')

async def test_initial_terminal_cursor_position():
    app = InputApp()
    async with app.run_test():
        assert app.cursor_position == Offset(21, 5)

async def test_terminal_cursor_position_update_on_cursor_move():
    app = InputApp()
    async with app.run_test():
        input_widget = app.query_one(Input)
        input_widget.action_cursor_left()
        input_widget.action_cursor_left()
        assert app.cursor_position == Offset(17, 5)