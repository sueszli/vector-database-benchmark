from textual.app import App, ComposeResult
from textual.widgets import Button

class ButtonIssue(App[None]):
    AUTO_FOCUS = None
    CSS = '\n    Button {\n        outline: white;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Button('Test')
if __name__ == '__main__':
    ButtonIssue().run()