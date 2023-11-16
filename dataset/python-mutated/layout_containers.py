"""
App to test layout containers.
"""
from typing import Iterable
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label

def sub_compose() -> Iterable[Widget]:
    if False:
        i = 10
        return i + 15
    yield Button.success('Accept')
    yield Button.error('Decline')
    yield Input()
    yield Label('\n\n'.join([str(n * 1000000) for n in range(10)]))

class MyApp(App[None]):
    CSS = '\n    Grid {\n        grid-size: 2 2;\n        grid-rows: 1fr;\n        grid-columns: 1fr;\n    }\n    Grid > Widget {\n        width: 100%;\n        height: 100%;\n    }\n    Input {\n        width: 80;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        with Grid():
            with Horizontal():
                yield from sub_compose()
            with HorizontalScroll():
                yield from sub_compose()
            with Vertical():
                yield from sub_compose()
            with VerticalScroll():
                yield from sub_compose()
app = MyApp()
if __name__ == '__main__':
    app.run()