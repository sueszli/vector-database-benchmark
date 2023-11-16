from __future__ import annotations
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.geometry import clamp
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Switch

class BitSwitch(Widget):
    """A Switch with a numeric label above it."""
    DEFAULT_CSS = '\n    BitSwitch {\n        layout: vertical;\n        width: auto;\n        height: auto;\n    }\n    BitSwitch > Label {\n        text-align: center;\n        width: 100%;\n    }\n    '

    class BitChanged(Message):
        """Sent when the 'bit' changes."""

        def __init__(self, bit: int, value: bool) -> None:
            if False:
                print('Hello World!')
            super().__init__()
            self.bit = bit
            self.value = value
    value = reactive(0)

    def __init__(self, bit: int) -> None:
        if False:
            i = 10
            return i + 15
        self.bit = bit
        super().__init__()

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Label(str(self.bit))
        yield Switch()

    def watch_value(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        'When the value changes we want to set the switch accordingly.'
        self.query_one(Switch).value = value

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if False:
            while True:
                i = 10
        'When the switch changes, notify the parent via a message.'
        event.stop()
        self.value = event.value
        self.post_message(self.BitChanged(self.bit, event.value))

class ByteInput(Widget):
    """A compound widget with 8 switches."""
    DEFAULT_CSS = '\n    ByteInput {\n        width: auto;\n        height: auto;\n        border: blank;\n        layout: horizontal;\n    }\n    ByteInput:focus-within {\n        border: heavy $secondary;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        for bit in reversed(range(8)):
            yield BitSwitch(bit)

class ByteEditor(Widget):
    DEFAULT_CSS = '\n    ByteEditor > Container {\n        height: 1fr;\n        align: center middle;\n    }\n    ByteEditor > Container.top {\n        background: $boost;\n    }\n    ByteEditor Input {\n        width: 16;\n    }\n    '
    value = reactive(0)

    def validate_value(self, value: int) -> int:
        if False:
            return 10
        'Ensure value is between 0 and 255.'
        return clamp(value, 0, 255)

    def compose(self) -> ComposeResult:
        if False:
            return 10
        with Container(classes='top'):
            yield Input(placeholder='byte')
        with Container():
            yield ByteInput()

    def on_bit_switch_bit_changed(self, event: BitSwitch.BitChanged) -> None:
        if False:
            while True:
                i = 10
        'When a switch changes, update the value.'
        value = 0
        for switch in self.query(BitSwitch):
            value |= switch.value << switch.bit
        self.query_one(Input).value = str(value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if False:
            for i in range(10):
                print('nop')
        'When the text changes, set the value of the byte.'
        try:
            self.value = int(event.value or '0')
        except ValueError:
            pass

    def watch_value(self, value: int) -> None:
        if False:
            print('Hello World!')
        'When self.value changes, update switches.'
        for switch in self.query(BitSwitch):
            with switch.prevent(BitSwitch.BitChanged):
                switch.value = bool(value & 1 << switch.bit)

class ByteInputApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield ByteEditor()
if __name__ == '__main__':
    app = ByteInputApp()
    app.run()