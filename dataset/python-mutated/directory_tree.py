from textual.app import App, ComposeResult
from textual.widgets import DirectoryTree

class DirectoryTreeApp(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield DirectoryTree('./')
if __name__ == '__main__':
    app = DirectoryTreeApp()
    app.run()