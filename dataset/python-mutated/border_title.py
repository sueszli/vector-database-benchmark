from textual.app import App, ComposeResult
from textual.widgets import Static
TEXT = 'I must not fear.\nFear is the mind-killer.\nFear is the little-death that brings total obliteration.\nI will face my fear.\nI will permit it to pass over me and through me.\nAnd when it has gone past, I will turn the inner eye to see its path.\nWhere the fear has gone there will be nothing. Only I will remain.'

class BorderTitleApp(App[None]):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        self.widget = Static(TEXT)
        yield self.widget

    def on_mount(self) -> None:
        if False:
            while True:
                i = 10
        self.widget.styles.background = 'darkblue'
        self.widget.styles.width = '50%'
        self.widget.styles.border = ('heavy', 'yellow')
        self.widget.border_title = 'Litany Against Fear'
        self.widget.border_subtitle = 'by Frank Herbert, in “Dune”'
        self.widget.styles.border_title_align = 'center'
if __name__ == '__main__':
    app = BorderTitleApp()
    app.run()