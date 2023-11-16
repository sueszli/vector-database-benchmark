from textual.app import App
from textual.containers import Grid
from textual.widgets import Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class AllTextStyleApp(App):
    CSS_PATH = 'text_style_all.tcss'

    def compose(self):
        if False:
            print('Hello World!')
        yield Grid(Label('none\n' + TEXT, id='lbl1'), Label('bold\n' + TEXT, id='lbl2'), Label('italic\n' + TEXT, id='lbl3'), Label('reverse\n' + TEXT, id='lbl4'), Label('strike\n' + TEXT, id='lbl5'), Label('underline\n' + TEXT, id='lbl6'), Label('bold italic\n' + TEXT, id='lbl7'), Label('reverse strike\n' + TEXT, id='lbl8'))
if __name__ == '__main__':
    app = AllTextStyleApp()
    app.run()