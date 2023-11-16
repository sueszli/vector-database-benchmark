from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Label

class VerticalRemoveApp(App[None]):
    CSS = '\n    Vertical {\n        border: round green;\n        height: auto;\n    }\n\n    Label {\n        border: round yellow;\n        background: red;\n        color: yellow;\n    }\n    '
    BINDINGS = [('a', 'add', 'Add'), ('d', 'del', 'Delete')]

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield Vertical()
        yield Footer()

    def action_add(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(Vertical).mount(Label('This is a test label'))

    def action_del(self) -> None:
        if False:
            while True:
                i = 10
        if self.query_one(Vertical).children:
            self.query_one(Vertical).children[-1].remove()
if __name__ == '__main__':
    VerticalRemoveApp().run()