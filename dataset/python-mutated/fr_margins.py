from textual.app import App, ComposeResult
from textual.widgets import Label
from textual.containers import Container

class TestApp(App):
    CSS = '\n    Container {\n        background: green 20%;\n        border: heavy green;\n        width: auto;\n        height: auto;\n        overflow: hidden;\n    }\n\n    Label {\n        background: green 20%;      \n        width: 1fr;\n        height: 1fr;\n        margin: 2 2;            \n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        with Container():
            yield Label('Hello')
            yield Label('World')
            yield Label('!!')
if __name__ == '__main__':
    app = TestApp()
    app.run()