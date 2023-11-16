from textual.app import App, ComposeResult
from textual.widgets import LoadingIndicator

class LoadingApp(App):

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield LoadingIndicator()
if __name__ == '__main__':
    app = LoadingApp()
    app.run()