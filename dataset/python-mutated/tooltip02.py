from textual.app import App, ComposeResult
from textual.widgets import Button
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.'

class TooltipApp(App):
    CSS = '\n    Screen {\n        align: center middle;\n    }\n    Tooltip {\n        padding: 2 4;\n        background: $primary;\n        color: auto 90%;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Button('Click me', variant='success')

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        self.query_one(Button).tooltip = TEXT
if __name__ == '__main__':
    app = TooltipApp()
    app.run()