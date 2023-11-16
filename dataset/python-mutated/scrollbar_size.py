from textual.app import App
from textual.containers import ScrollableContainer
from textual.widgets import Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.\n'

class ScrollbarApp(App):
    CSS_PATH = 'scrollbar_size.tcss'

    def compose(self):
        if False:
            i = 10
            return i + 15
        yield ScrollableContainer(Label(TEXT * 5), classes='panel')
if __name__ == '__main__':
    app = ScrollbarApp()
    app.run()