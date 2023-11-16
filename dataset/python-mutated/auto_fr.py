from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.widgets import Label

class FRApp(App):
    CSS = '    \n    Screen {\n        align: center middle;\n        border: solid cyan;\n    }\n\n    #container {  \n        width: 30;      \n        height: auto;\n        border: solid green;\n        overflow-y: auto;\n    }\n\n    #child {\n        height: 1fr;\n        border: solid red;       \n    }\n\n    #bottom {\n        margin: 1 2;    \n        background: $primary;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        with Widget(id='container'):
            yield Label('Hello one line', id='top')
            yield Widget(id='child')
            yield Label('Two\nLines with 1x2 margin', id='bottom')
if __name__ == '__main__':
    app = FRApp()
    app.run()