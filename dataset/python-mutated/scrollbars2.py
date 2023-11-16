from textual.app import App
from textual.widgets import Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.\n'

class Scrollbar2App(App):
    CSS_PATH = 'scrollbars2.tcss'

    def compose(self):
        if False:
            while True:
                i = 10
        yield Label(TEXT * 10)
if __name__ == '__main__':
    app = Scrollbar2App()
    app.run()