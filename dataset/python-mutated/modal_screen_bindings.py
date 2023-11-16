from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, Input

class Dialog(ModalScreen):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Label('Dialog')
        yield Input()
        yield Button('OK', id='ok')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if False:
            i = 10
            return i + 15
        if event.button.id == 'ok':
            self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if False:
            return 10
        self.app.pop_screen()

class ModalApp(App):
    BINDINGS = [('enter', 'open_dialog', 'Open Dialog')]

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield Label('Hello')
        yield Footer()

    def action_open_dialog(self) -> None:
        if False:
            while True:
                i = 10
        self.push_screen(Dialog())
if __name__ == '__main__':
    app = ModalApp()
    app.run()