from textual.app import App, ComposeResult
from textual.widgets import Button

class ButtonApp(App):
    CSS = '\n    Button {\n        height: 9;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Button('Hello')
        yield Button('Hello\nWorld !!')
if __name__ == '__main__':
    app = ButtonApp()
    app.run()