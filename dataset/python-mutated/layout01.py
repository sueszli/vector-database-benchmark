from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Placeholder

class Header(Placeholder):
    pass

class Footer(Placeholder):
    pass

class TweetScreen(Screen):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Header(id='Header')
        yield Footer(id='Footer')

class LayoutApp(App):

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        self.push_screen(TweetScreen())
if __name__ == '__main__':
    app = LayoutApp()
    app.run()