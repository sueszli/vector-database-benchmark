from textual.app import App, ComposeResult
from textual import events
from textual.widgets import RichLog

class PrintLogger(RichLog):
    """A RichLog which captures printed text."""

    def on_print(self, event: events.Print) -> None:
        if False:
            return 10
        self.write(event.text)

class CaptureApp(App):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield PrintLogger()

    def on_mount(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(RichLog).write('RichLog')
        self.query_one(RichLog).begin_capture_print()
        print('This will be captured!')
        self.query_one(RichLog).end_capture_print()
        print('This will *not* be captured')
if __name__ == '__main__':
    app = CaptureApp()
    app.run()