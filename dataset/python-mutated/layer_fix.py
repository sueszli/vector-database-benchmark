from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Header, Footer, Label
from textual.binding import Binding

class Dialog(Vertical):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        'Compose the child widgets.'
        yield Label('This should not cause a scrollbar to appear')

class DialogIssueApp(App[None]):
    CSS = '\n    Screen {\n        layers: base dialog;\n    }\n\n    .hidden {\n        display: none;\n    }\n\n    Dialog {\n        align: center middle;\n        border: round red;\n        width: 50%;\n        height: 50%;\n        layer: dialog;\n        offset: 50% 50%;\n    }\n    '
    BINDINGS = [Binding('d', 'dialog', 'Toggle the dialog')]

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Header()
        yield Vertical()
        yield Dialog(classes='hidden')
        yield Footer()

    def action_dialog(self) -> None:
        if False:
            while True:
                i = 10
        self.query_one(Dialog).toggle_class('hidden')
if __name__ == '__main__':
    DialogIssueApp().run()