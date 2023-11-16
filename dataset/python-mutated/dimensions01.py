from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class DimensionsApp(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        self.widget = Static(TEXT)
        yield self.widget

    def on_mount(self) -> None:
        if False:
            return 10
        self.widget.styles.background = 'purple'
        self.widget.styles.width = 30
        self.widget.styles.height = 10
if __name__ == '__main__':
    app = DimensionsApp()
    app.run()