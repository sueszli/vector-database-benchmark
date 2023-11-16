from textual import events
from textual.app import App, ComposeResult
from textual.widgets import RichLog, Static

class Ball(Static):
    pass

class MouseApp(App):
    CSS_PATH = 'mouse01.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield RichLog()
        yield Ball('Textual')

    def on_mouse_move(self, event: events.MouseMove) -> None:
        if False:
            return 10
        self.screen.query_one(RichLog).write(event)
        self.query_one(Ball).offset = event.screen_offset - (8, 2)
if __name__ == '__main__':
    app = MouseApp()
    app.run()