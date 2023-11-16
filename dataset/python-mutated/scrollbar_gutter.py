from textual.app import App
from textual.widgets import Static
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class ScrollbarGutterApp(App):
    CSS_PATH = 'scrollbar_gutter.tcss'

    def compose(self):
        if False:
            i = 10
            return i + 15
        yield Static(TEXT, id='text-box')
if __name__ == '__main__':
    app = ScrollbarGutterApp()
    app.run()