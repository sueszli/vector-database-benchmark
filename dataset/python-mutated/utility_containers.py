from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

class UtilityContainersExample(App):
    CSS_PATH = 'utility_containers.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Horizontal(Vertical(Static('One'), Static('Two'), classes='column'), Vertical(Static('Three'), Static('Four'), classes='column'))
if __name__ == '__main__':
    app = UtilityContainersExample()
    app.run()