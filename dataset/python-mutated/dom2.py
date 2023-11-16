from textual.app import App, ComposeResult
from textual.widgets import Header, Footer

class ExampleApp(App):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield Footer()
if __name__ == '__main__':
    app = ExampleApp()
    app.run()