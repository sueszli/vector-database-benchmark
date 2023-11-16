from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Label

class MyWidget(Widget):
    DEFAULT_CSS = '\n    MyWidget {\n        height: auto;\n        border: magenta;\n    }\n    Label {\n        border: solid green;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield Label('foo')
        yield Label('bar')

    def on_mount(self) -> None:
        if False:
            i = 10
            return i + 15
        self.log(self.app.stylesheet.css)

class MyApp(App):

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield MyWidget()
        yield MyWidget()
        yield Label('I should not be styled')
if __name__ == '__main__':
    app = MyApp()
    app.run()