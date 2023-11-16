from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Placeholder

class Header(Placeholder):
    DEFAULT_CSS = '\n    Header {\n        height: 3;\n        dock: top;\n    }\n    '

class Footer(Placeholder):
    DEFAULT_CSS = '\n    Footer {\n        height: 3;\n        dock: bottom;\n    }\n    '

class TweetScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header(id='Header')
        yield Footer(id='Footer')

class LayoutApp(App):

    def on_ready(self) -> None:
        if False:
            print('Hello World!')
        self.push_screen(TweetScreen())
if __name__ == '__main__':
    app = LayoutApp()
    app.run()