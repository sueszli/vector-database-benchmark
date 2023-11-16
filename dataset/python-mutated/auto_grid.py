from textual.app import App, ComposeResult
from textual.widgets import Label, Input
from textual.containers import Container

class GridApp(App):
    CSS = '\n    Screen {\n        align: center middle;\n    }\n    Container {\n        layout: grid;\n        grid-size: 2;\n        grid-columns: auto 1fr;\n        grid-rows: auto;\n        height:auto;\n        border: solid green;\n    }\n    \n    #c2 Label {\n        min-width: 20;\n    }\n\n    #c3 Label {\n        max-width: 30;\n    }\n\n    '
    AUTO_FOCUS = None

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        with Container(id='c1'):
            yield Label('foo')
            yield Input()
            yield Label('Longer label')
            yield Input()
        with Container(id='c2'):
            yield Label('foo')
            yield Input()
            yield Label('Longer label')
            yield Input()
        with Container(id='c3'):
            yield Label('foo bar ' * 10)
            yield Input()
            yield Label('Longer label')
            yield Input()
if __name__ == '__main__':
    app = GridApp()
    app.run()