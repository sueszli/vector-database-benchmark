from textual.app import App, ComposeResult, RenderResult
from textual.widget import Widget

class Hello(Widget):
    """Display a greeting."""

    def render(self) -> RenderResult:
        if False:
            return 10
        return 'Hello, [b]World[/b]!'

class CustomApp(App):
    CSS_PATH = 'hello02.tcss'

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Hello()
if __name__ == '__main__':
    app = CustomApp()
    app.run()