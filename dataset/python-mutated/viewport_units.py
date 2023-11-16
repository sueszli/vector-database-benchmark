from textual.app import App, ComposeResult
from textual.widgets import Static

class ViewportUnits(App):
    CSS = 'Static {width: 100vw; height: 100vh; border: solid cyan;} '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Static('Hello, world!')
app = ViewportUnits()
if __name__ == '__main__':
    app.run()