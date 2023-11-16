from __future__ import annotations
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Input, Label, Switch

class BitSwitch(Widget):
    """A Switch with a numeric label above it."""
    DEFAULT_CSS = '\n    BitSwitch {\n        layout: vertical;\n        width: auto;\n        height: auto;\n    }\n    BitSwitch > Label {\n        text-align: center;\n        width: 100%;\n    }\n    '

    def __init__(self, bit: int) -> None:
        if False:
            i = 10
            return i + 15
        self.bit = bit
        super().__init__()

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Label(str(self.bit))
        yield Switch()

class ByteInput(Widget):
    """A compound widget with 8 switches."""
    DEFAULT_CSS = '\n    ByteInput {\n        width: auto;\n        height: auto;\n        border: blank;\n        layout: horizontal;\n    }\n    ByteInput:focus-within {\n        border: heavy $secondary;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        for bit in reversed(range(8)):
            yield BitSwitch(bit)

class ByteEditor(Widget):
    DEFAULT_CSS = '\n    ByteEditor > Container {\n        height: 1fr;\n        align: center middle;\n    }\n    ByteEditor > Container.top {\n        background: $boost;\n    }\n    ByteEditor Input {\n        width: 16;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        with Container(classes='top'):
            yield Input(placeholder='byte')
        with Container():
            yield ByteInput()

class ByteInputApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield ByteEditor()
if __name__ == '__main__':
    app = ByteInputApp()
    app.run()