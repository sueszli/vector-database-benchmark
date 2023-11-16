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
            for i in range(10):
                print('nop')
        self.query_one(RichLog).write(event)
if __name__ == '__main__':
    app = InputApp()
    app.run()