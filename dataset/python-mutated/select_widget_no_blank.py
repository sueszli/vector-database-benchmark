from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Header, Select
LINES = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.'.splitlines()
ALTERNATE_LINES = 'Twinkle, twinkle, little star,\nHow I wonder what you are!\nUp above the world so high,\nLike a diamond in the sky.\nTwinkle, twinkle, little star,\nHow I wonder what you are!'.splitlines()

class SelectApp(App):
    CSS_PATH = 'select.tcss'
    BINDINGS = [('s', 'swap', 'Swap Select options')]

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Header()
        yield Select(zip(LINES, LINES), allow_blank=False)

    @on(Select.Changed)
    def select_changed(self, event: Select.Changed) -> None:
        if False:
            while True:
                i = 10
        self.title = str(event.value)

    def action_swap(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.query_one(Select).set_options(zip(ALTERNATE_LINES, ALTERNATE_LINES))
if __name__ == '__main__':
    app = SelectApp()
    app.run()