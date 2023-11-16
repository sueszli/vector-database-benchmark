from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

class AutoApp(App):
    """The innermost container should push its parents outwards, to fill the screen."""
    CSS = '\n    #outer {\n        background: blue;\n        height: auto;\n        border: solid white;\n    } \n\n    #inner {\n        background: green;\n        height: auto;\n        border: solid yellow;\n    }\n\n    #innermost {\n        background: cyan;\n        height: 1fr;\n        color: auto;        \n    }\n            \n    '

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        with Vertical(id='outer'):
            with Vertical(id='inner'):
                with Vertical(id='innermost'):
                    yield Static('Hello\nWorld!\nfoo', id='helloworld')
if __name__ == '__main__':
    app = AutoApp()
    app.run()