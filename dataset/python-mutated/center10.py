from textual.app import App, ComposeResult
from textual.containers import Center
from textual.widgets import Static

class CenterApp(App):
    """How to center things."""
    CSS = '\n    Screen {\n        align: center middle;\n    }\n\n    .words {\n        background: blue 50%;\n        border: wide white;\n        width: auto;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        with Center():
            yield Static('How about a nice game', classes='words')
        with Center():
            yield Static('of chess?', classes='words')
if __name__ == '__main__':
    app = CenterApp()
    app.run()