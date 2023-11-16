from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class PaddingApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        self.widget = Static(TEXT)
        yield self.widget

    def on_mount(self) -> None:
        if False:
            return 10
        self.widget.styles.background = 'purple'
        self.widget.styles.width = 30
        self.widget.styles.padding = 2
if __name__ == '__main__':
    app = PaddingApp()
    app.run()