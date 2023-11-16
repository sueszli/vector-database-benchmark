"""
App to test alignment containers.
"""
from textual.app import App, ComposeResult
from textual.containers import Center, Middle
from textual.widgets import Button

class AlignContainersApp(App[None]):
    CSS = '\n    Center {\n        tint: $primary 10%;\n    }\n    Middle {\n        tint: $secondary 10%;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        with Center():
            yield Button.success('center')
        with Middle():
            yield Button.error('middle')
app = AlignContainersApp()
if __name__ == '__main__':
    app.run()