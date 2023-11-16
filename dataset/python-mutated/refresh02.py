from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input

class Name(Widget):
    """Generates a greeting."""
    who = reactive('name', layout=True)

    def render(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'Hello, {self.who}!'

class WatchApp(App):
    CSS_PATH = 'refresh02.tcss'

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Input(placeholder='Enter your name')
        yield Name()

    def on_input_changed(self, event: Input.Changed) -> None:
        if False:
            print('Hello World!')
        self.query_one(Name).who = event.value
if __name__ == '__main__':
    app = WatchApp()
    app.run()