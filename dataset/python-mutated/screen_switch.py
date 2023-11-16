from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Header, Footer

class ScreenA(Screen):
    BINDINGS = [('b', 'switch_to_b', 'Switch to screen B')]

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Header()
        yield Static('A')
        yield Footer()

    def action_switch_to_b(self):
        if False:
            print('Hello World!')
        self.app.switch_screen(ScreenB())

class ScreenB(Screen):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Header()
        yield Static('B')
        yield Footer()

class ModalApp(App):
    BINDINGS = [('a', 'push_a', 'Push screen A')]

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Header()
        yield Footer()

    def action_push_a(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.push_screen(ScreenA())
if __name__ == '__main__':
    app = ModalApp()
    app.run()