from textual.app import App, ComposeResult
from textual.widgets import Header

class HeaderApp(App):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Header()

    def on_mount(self) -> None:
        if False:
            print('Hello World!')
        self.title = 'Header Application'
        self.sub_title = 'With title and sub-title'
if __name__ == '__main__':
    app = HeaderApp()
    app.run()