from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Header, Footer, Label
from textual.containers import VerticalScroll, Container

class Overlay(Container):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Label('This should float over the top')

class Body1(VerticalScroll):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Label("My God! It's full of stars! " * 300)

class Body2(VerticalScroll):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Label("My God! It's full of stars! " * 300)

class Good(Screen):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield Overlay()
        yield Body1()
        yield Footer()

class Bad(Screen):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Overlay()
        yield Header()
        yield Body2()
        yield Footer()

class Layers(App[None]):
    CSS = '\n    Screen {\n        layers: base higher;\n    }\n\n    Overlay {\n        layer: higher;\n        dock: top;\n        width: auto;\n        height: auto;\n        padding: 2;\n        border: solid yellow;\n        background: red;\n        color: yellow;\n    }\n\n    Body2 {\n        background: green;\n    }\n    '
    SCREENS = {'good': Good, 'bad': Bad}
    BINDINGS = [('t', 'toggle', 'Toggle Screen')]

    def on_mount(self):
        if False:
            return 10
        self.push_screen('good')

    def action_toggle(self):
        if False:
            while True:
                i = 10
        self.switch_screen('bad' if self.screen.__class__.__name__ == 'Good' else 'good')
if __name__ == '__main__':
    Layers().run()