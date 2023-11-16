from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Placeholder

class Header(Placeholder):
    DEFAULT_CSS = '\n    Header {\n        height: 3;\n        dock: top;\n    }\n    '

class Footer(Placeholder):
    DEFAULT_CSS = '\n    Footer {\n        height: 3;\n        dock: bottom;\n    }\n    '

class ColumnsContainer(Placeholder):
    DEFAULT_CSS = '\n    ColumnsContainer {\n        width: 1fr;\n        height: 1fr;\n        border: solid white;\n    }\n    '

class TweetScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield Header(id='Header')
        yield Footer(id='Footer')
        yield ColumnsContainer(id='Columns')

class LayoutApp(App):

    def on_ready(self) -> None:
        if False:
            while True:
                i = 10
        self.push_screen(TweetScreen())
if __name__ == '__main__':
    app = LayoutApp()
    app.run()