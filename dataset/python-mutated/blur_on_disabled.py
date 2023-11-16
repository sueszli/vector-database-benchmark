from textual.app import App, ComposeResult
from textual.widgets import Input

class BlurApp(App):
    BINDINGS = [('f3', 'disable')]

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Input()

    def on_ready(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(Input).focus()

    def action_disable(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(Input).disabled = True
if __name__ == '__main__':
    app = BlurApp()
    app.run()