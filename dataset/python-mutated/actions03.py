from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = "\n[b]Set your background[/b]\n[@click=set_background('red')]Red[/]\n[@click=set_background('green')]Green[/]\n[@click=set_background('blue')]Blue[/]\n"

class ActionsApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Static(TEXT)

    def action_set_background(self, color: str) -> None:
        if False:
            return 10
        self.screen.styles.background = color
if __name__ == '__main__':
    app = ActionsApp()
    app.run()