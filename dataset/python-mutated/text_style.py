from textual.app import App
from textual.widgets import Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class TextStyleApp(App):
    CSS_PATH = 'text_style.tcss'

    def compose(self):
        if False:
            for i in range(10):
                print('nop')
        yield Label(TEXT, id='lbl1')
        yield Label(TEXT, id='lbl2')
        yield Label(TEXT, id='lbl3')
if __name__ == '__main__':
    app = TextStyleApp()
    app.run()