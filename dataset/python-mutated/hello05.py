from itertools import cycle
from textual.app import App, ComposeResult
from textual.widgets import Static
hellos = cycle(['Hola', 'Bonjour', 'Guten tag', 'Salve', 'Nǐn hǎo', 'Olá', 'Asalaam alaikum', 'Konnichiwa', 'Anyoung haseyo', 'Zdravstvuyte', 'Hello'])

class Hello(Static):
    """Display a greeting."""

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.action_next_word()

    def action_next_word(self) -> None:
        if False:
            return 10
        'Get a new hello and update the content area.'
        hello = next(hellos)
        self.update(f"[@click='next_word']{hello}[/], [b]World[/b]!")

class CustomApp(App):
    CSS_PATH = 'hello05.tcss'

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Hello()
if __name__ == '__main__':
    app = CustomApp()
    app.run()