from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static

class ScreenSplitApp(App[None]):
    CSS = '\n    Horizontal {\n        width: 1fr;        \n    }\n\n    Vertical {\n        width: 1fr;\n        background: blue;\n        min-width: 20;\n    }\n\n    #scroll1 {\n        width: 1fr;\n        background: $panel;\n    }\n\n    #scroll2 {\n        width: 2fr;\n        background: $panel;\n    }\n\n    Static {\n        width: 1fr;\n        content-align: center middle;\n        background: $boost;\n    }\n\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Header()
        with Horizontal():
            yield Vertical()
            with VerticalScroll(id='scroll1'):
                for n in range(100):
                    yield Static(f'This is content number {n}')
            with VerticalScroll(id='scroll2'):
                for n in range(100):
                    yield Static(f'This is content number {n}')
        yield Footer()
if __name__ == '__main__':
    ScreenSplitApp().run()