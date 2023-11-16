from textual.app import App
from textual.widgets import Label
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class OutlineBorderApp(App):
    CSS_PATH = 'outline_vs_border.tcss'

    def compose(self):
        if False:
            print('Hello World!')
        yield Label(TEXT, classes='outline')
        yield Label(TEXT, classes='border')
        yield Label(TEXT, classes='outline border')
if __name__ == '__main__':
    app = OutlineBorderApp()
    app.run()