from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog

class InputApp(App):
    """App to display key events."""

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield RichLog()

    def on_key(self, event: events.Key) -> None:
        if False:
            while True:
                i = 10
        self.query_one(RichLog).write(event)

    def key_space(self) -> None:
        if False:
            i = 10
            return i + 15
        self.bell()
if __name__ == '__main__':
    app = InputApp()
    app.run()