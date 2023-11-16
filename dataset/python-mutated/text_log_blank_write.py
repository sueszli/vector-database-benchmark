from textual.app import App, ComposeResult
from textual.widgets import RichLog

class RichLogApp(App):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield RichLog()

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        tl = self.query_one(RichLog)
        tl.write('Hello')
        tl.write('')
        tl.write('World')
if __name__ == '__main__':
    app = RichLogApp()
    app.run()