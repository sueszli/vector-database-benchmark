from __future__ import annotations
from typing import Type
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.widgets import Header, Footer, Label
from textual.binding import Binding

class GridHeightAuto(App[None]):
    CSS = '\n    #test-area {\n        \n        border: solid red;\n        height: auto;\n    }\n\n    Grid {\n        grid-size: 3;\n        # grid-rows: auto;\n    }\n    '
    BINDINGS = [Binding('g', 'grid', 'Grid'), Binding('v', 'vertical', 'Vertical'), Binding('h', 'horizontal', 'Horizontal'), Binding('c', 'container', 'Container')]

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Header()
        yield Vertical(Label('Select a container to test (see footer)'), id='sandbox')
        yield Footer()

    def build(self, out_of: Type[Container | Grid | Horizontal | Vertical]) -> None:
        if False:
            while True:
                i = 10
        self.query('#sandbox > *').remove()
        self.query_one('#sandbox', Vertical).mount(Label('Here is some text before the grid'), out_of(*[Label(f'Cell #{n}') for n in range(9)], id='test-area'), Label('Here is some text after the grid'))

    def action_grid(self):
        if False:
            return 10
        self.build(Grid)

    def action_vertical(self):
        if False:
            i = 10
            return i + 15
        self.build(Vertical)

    def action_horizontal(self):
        if False:
            i = 10
            return i + 15
        self.build(Horizontal)

    def action_container(self):
        if False:
            print('Hello World!')
        self.build(Container)
if __name__ == '__main__':
    GridHeightAuto().run()