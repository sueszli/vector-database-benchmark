from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button

class WidgetDisableTestApp(App[None]):
    CSS = '\n    Horizontal {\n        height: auto;\n    }\n\n    Button {\n        width: 1fr;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        for _ in range(4):
            with Horizontal():
                yield Button()
                yield Button(variant='primary')
                yield Button(variant='success')
                yield Button(variant='warning')
                yield Button(variant='error')
            with Horizontal(disabled=True):
                yield Button()
                yield Button(variant='primary')
                yield Button(variant='success')
                yield Button(variant='warning')
                yield Button(variant='error')
if __name__ == '__main__':
    WidgetDisableTestApp().run()