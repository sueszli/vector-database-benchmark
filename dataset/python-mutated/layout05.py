from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import Screen
from textual.widgets import Placeholder

class Header(Placeholder):
    DEFAULT_CSS = '\n    Header {\n        height: 3;\n        dock: top;\n    }\n    '

class Footer(Placeholder):
    DEFAULT_CSS = '\n    Footer {\n        height: 3;\n        dock: bottom;\n    }\n    '

class Tweet(Placeholder):
    pass

class Column(VerticalScroll):

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        for tweet_no in range(1, 20):
            yield Tweet(id=f'Tweet{tweet_no}')

class TweetScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Header(id='Header')
        yield Footer(id='Footer')
        with HorizontalScroll():
            yield Column()
            yield Column()
            yield Column()
            yield Column()

class LayoutApp(App):

    def on_ready(self) -> None:
        if False:
            return 10
        self.push_screen(TweetScreen())
if __name__ == '__main__':
    app = LayoutApp()
    app.run()