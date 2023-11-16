from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = "\n[b]Set your background[/b]\n[@click=set_background('red')]Red[/]\n[@click=set_background('green')]Green[/]\n[@click=set_background('blue')]Blue[/]\n"

class ActionsApp(App):
    BINDINGS = [('r', "set_background('red')", 'Red'), ('g', "set_background('green')", 'Green'), ('b', "set_background('blue')", 'Blue')]

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Static(TEXT)

    def action_set_background(self, color: str) -> None:
        if False:
            return 10
        self.screen.styles.background = color
if __name__ == '__main__':
    app = ActionsApp()
    app.run()