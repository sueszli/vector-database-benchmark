from textual.app import App, ComposeResult
from textual.widgets import Static
QUOTE = 'Could not find you in Seattle and no terminal is in operation at your classified address.'

class CenterApp(App):
    """How to center things."""
    CSS = '\n    Screen {\n        align: center middle;\n    }\n\n    #hello {\n        background: blue 50%;\n        border: wide white;\n        width: 40;\n        text-align: center;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Static(QUOTE, id='hello')
if __name__ == '__main__':
    app = CenterApp()
    app.run()