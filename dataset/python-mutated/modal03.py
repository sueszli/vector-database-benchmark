from textual.app import App, ComposeResult
from textual.containers import Grid
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class QuitScreen(ModalScreen[bool]):
    """Screen with a dialog to quit."""

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Grid(Label('Are you sure you want to quit?', id='question'), Button('Quit', variant='error', id='quit'), Button('Cancel', variant='primary', id='cancel'), id='dialog')

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if False:
            i = 10
            return i + 15
        if event.button.id == 'quit':
            self.dismiss(True)
        else:
            self.dismiss(False)

class ModalApp(App):
    """An app with a modal dialog."""
    CSS_PATH = 'modal01.tcss'
    BINDINGS = [('q', 'request_quit', 'Quit')]

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Header()
        yield Label(TEXT * 8)
        yield Footer()

    def action_request_quit(self) -> None:
        if False:
            print('Hello World!')
        'Action to display the quit dialog.'

        def check_quit(quit: bool) -> None:
            if False:
                print('Hello World!')
            'Called when QuitScreen is dismissed.'
            if quit:
                self.exit()
        self.push_screen(QuitScreen(), check_quit)
if __name__ == '__main__':
    app = ModalApp()
    app.run()