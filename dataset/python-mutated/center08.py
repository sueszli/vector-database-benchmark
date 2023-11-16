from textual.app import App, ComposeResult
from textual.widgets import Static

class CenterApp(App):
    """How to center things."""
    CSS = '\n    .words {\n        background: blue 50%;\n        border: wide white;\n        width: auto;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Static('How about a nice game', classes='words')
        yield Static('of chess?', classes='words')
if __name__ == '__main__':
    app = CenterApp()
    app.run()