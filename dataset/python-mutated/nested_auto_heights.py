from __future__ import annotations
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

class NestedAutoApp(App[None]):
    CSS = '\n    Screen {\n        background: red;\n    }\n\n    #my-static-container {\n        border: heavy lightgreen;\n        background: green;\n        height: auto;\n        max-height: 10;\n    }\n\n    #my-static-wrapper {\n        border: heavy lightblue;\n        background: blue;\n        width: auto;\n        height: auto;\n    }\n\n    #my-static {\n        border: heavy gray;\n        background: black;\n        width: auto;\n        height: auto;\n    }\n    '
    BINDINGS = [('1', '1', '1'), ('2', '2', '2'), ('q', 'quit', 'Quit')]

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        self._static = Static('', id='my-static')
        yield VerticalScroll(VerticalScroll(self._static, id='my-static-wrapper'), id='my-static-container')

    def action_1(self) -> None:
        if False:
            print('Hello World!')
        self._static.update('\n'.join((f'Lorem {i} Ipsum {i} Sit {i}' for i in range(1, 21))))

    def action_2(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._static.update('JUST ONE LINE')
if __name__ == '__main__':
    app = NestedAutoApp()
    app.run()