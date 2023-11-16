from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button

class OnDecoratorApp(App):
    CSS_PATH = 'on_decorator.tcss'

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        'Three buttons.'
        yield Button('Bell', id='bell')
        yield Button('Toggle dark', classes='toggle dark')
        yield Button('Quit', id='quit')

    @on(Button.Pressed, '#bell')
    def play_bell(self):
        if False:
            while True:
                i = 10
        'Called when the bell button is pressed.'
        self.bell()

    @on(Button.Pressed, '.toggle.dark')
    def toggle_dark(self):
        if False:
            i = 10
            return i + 15
        "Called when the 'toggle dark' button is pressed."
        self.dark = not self.dark

    @on(Button.Pressed, '#quit')
    def quit(self):
        if False:
            i = 10
            return i + 15
        'Called when the quit button is pressed.'
        self.exit()
if __name__ == '__main__':
    app = OnDecoratorApp()
    app.run()