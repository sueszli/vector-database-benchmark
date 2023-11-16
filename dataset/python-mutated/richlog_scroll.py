from textual.app import App, ComposeResult
from textual.widgets import RichLog
from textual.containers import Horizontal

class RichLogScrollApp(App):
    CSS = '\n    RichLog{\n        width: 1fr;\n        height: 10;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        with Horizontal():
            yield RichLog(id='richlog1', auto_scroll=False)
            yield RichLog(id='richlog2', auto_scroll=True)
            yield RichLog(id='richlog3', auto_scroll=True)

    def on_ready(self) -> None:
        if False:
            print('Hello World!')
        lines = [f'Line {n}' for n in range(20)]
        for line in lines:
            self.query_one('#richlog1', RichLog).write(line)
        for line in lines:
            self.query_one('#richlog2', RichLog).write(line)
        for line in lines:
            self.query_one('#richlog3', RichLog).write(line, scroll_end=False)
if __name__ == '__main__':
    app = RichLogScrollApp()
    app.run()