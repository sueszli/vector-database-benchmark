from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.screen import Screen
from textual.widgets import Placeholder

class Header(Placeholder):
    DEFAULT_CSS = '\n    Header {\n        height: 3;\n        dock: top;\n    }\n    '

class Footer(Placeholder):
    DEFAULT_CSS = '\n    Footer {\n        height: 3;\n        dock: bottom;\n    }\n    '

class Tweet(Placeholder):
    DEFAULT_CSS = '\n    Tweet {\n        height: 5;\n        width: 1fr;\n        border: tall $background;\n    }\n    '

class Column(VerticalScroll):
    DEFAULT_CSS = '\n    Column {\n        height: 1fr;\n        width: 32;\n        margin: 0 2;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        for tweet_no in range(1, 20):
            yield Tweet(id=f'Tweet{tweet_no}')

class TweetScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
        self.push_screen(TweetScreen())
if __name__ == '__main__':
    app = LayoutApp()
    app.run()