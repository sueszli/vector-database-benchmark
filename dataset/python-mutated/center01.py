from textual.app import App, ComposeResult
from textual.widgets import Static

class CenterApp(App):
    """How to center things."""
    CSS = '\n    Screen {\n        align: center middle;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Static('Hello, World!')
if __name__ == '__main__':
    app = CenterApp()
    app.run()