from textual.app import App, ComposeResult
from textual.widgets import Button
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.'

class TooltipApp(App):
    CSS = '\n    Screen {\n        align: center middle;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Button('Click me', variant='success')

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(Button).tooltip = TEXT
if __name__ == '__main__':
    app = TooltipApp()
    app.run()