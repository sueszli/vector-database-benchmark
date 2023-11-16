from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Input, Label

class InputWithLabel(Widget):
    """An input with a label."""
    DEFAULT_CSS = '\n    InputWithLabel {\n        layout: horizontal;\n        height: auto;\n    }\n    InputWithLabel Label {\n        padding: 1;\n        width: 12;\n        text-align: right;\n    }\n    InputWithLabel Input {\n        width: 1fr;\n    }\n    '

    def __init__(self, input_label: str) -> None:
        if False:
            return 10
        self.input_label = input_label
        super().__init__()

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Label(self.input_label)
        yield Input()

class CompoundApp(App):
    CSS = '\n    Screen {\n        align: center middle;\n    }\n    InputWithLabel {\n        width: 80%;\n        margin: 1;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield InputWithLabel('First Name')
        yield InputWithLabel('Last Name')
        yield InputWithLabel('Email')
if __name__ == '__main__':
    app = CompoundApp()
    app.run()