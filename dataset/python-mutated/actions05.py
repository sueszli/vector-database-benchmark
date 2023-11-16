from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = "\n[b]Set your background[/b]\n[@click=set_background('cyan')]Cyan[/]\n[@click=set_background('magenta')]Magenta[/]\n[@click=set_background('yellow')]Yellow[/]\n"

class ColorSwitcher(Static):

    def action_set_background(self, color: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.styles.background = color

class ActionsApp(App):
    CSS_PATH = 'actions05.tcss'
    BINDINGS = [('r', "set_background('red')", 'Red'), ('g', "set_background('green')", 'Green'), ('b', "set_background('blue')", 'Blue')]

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield ColorSwitcher(TEXT)
        yield ColorSwitcher(TEXT)

    def action_set_background(self, color: str) -> None:
        if False:
            print('Hello World!')
        self.screen.styles.background = color
if __name__ == '__main__':
    app = ActionsApp()
    app.run()