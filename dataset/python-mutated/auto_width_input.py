from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input

class InputWidthAutoApp(App[None]):
    CSS = '\n    Input.auto {\n        width: auto;\n        max-width: 100%;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Header()
        yield Input(placeholder='This has auto width', classes='auto')
        yield Footer()
if __name__ == '__main__':
    InputWidthAutoApp().run()