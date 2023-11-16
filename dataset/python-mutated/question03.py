from textual.app import App, ComposeResult
from textual.widgets import Label, Button

class QuestionApp(App[str]):
    CSS = '\n    Screen {\n        layout: grid;\n        grid-size: 2;\n        grid-gutter: 2;\n        padding: 2;\n    }\n    #question {\n        width: 100%;\n        height: 100%;\n        column-span: 2;\n        content-align: center bottom;\n        text-style: bold;\n    }\n\n    Button {\n        width: 100%;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Label('Do you love Textual?', id='question')
        yield Button('Yes', id='yes', variant='primary')
        yield Button('No', id='no', variant='error')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.exit(event.button.id)
if __name__ == '__main__':
    app = QuestionApp()
    reply = app.run()
    print(reply)