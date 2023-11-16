from textual.app import App, ComposeResult
from textual.widgets import Collapsible, Label

class CollapsibleApp(App[None]):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        with Collapsible(collapsed=False):
            with Collapsible():
                yield Label('Hello, world.')
if __name__ == '__main__':
    app = CollapsibleApp()
    app.run()