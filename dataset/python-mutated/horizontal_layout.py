from textual.app import App, ComposeResult
from textual.widgets import Static

class HorizontalLayoutExample(App):
    CSS_PATH = 'horizontal_layout.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Static('One', classes='box')
        yield Static('Two', classes='box')
        yield Static('Three', classes='box')
if __name__ == '__main__':
    app = HorizontalLayoutExample()
    app.run()