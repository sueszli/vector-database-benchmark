from textual.app import App, ComposeResult
from textual.widgets import Button, Input

class PreventApp(App):
    """Demonstrates `prevent` context manager."""

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Input()
        yield Button('Clear', id='clear')

    def on_button_pressed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Clear the text input.'
        input = self.query_one(Input)
        with input.prevent(Input.Changed):
            input.value = ''

    def on_input_changed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Called as the user types.'
        self.bell()
if __name__ == '__main__':
    app = PreventApp()
    app.run()