from textual.app import App, ComposeResult
from textual.widgets import Static

class GridLayoutExample(App):
    CSS_PATH = 'grid_layout7_gutter.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Static('One', classes='box')
        yield Static('Two', classes='box')
        yield Static('Three', classes='box')
        yield Static('Four', classes='box')
        yield Static('Five', classes='box')
        yield Static('Six', classes='box')
if __name__ == '__main__':
    app = GridLayoutExample()
    app.run()