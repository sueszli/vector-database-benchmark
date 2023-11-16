from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

class HeightApp(App[None]):
    CSS = '\n    Horizontal {\n        border: solid red;\n        height: auto;\n    }\n\n    Static {\n        border: solid green;\n        width: auto;\n    }\n\n    #fill_parent {\n        height: 100%;\n    }\n\n    #static {\n        height: 16;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Horizontal(Static('As tall as container', id='fill_parent'), Static('This has default\nheight\nbut a\nfew lines'), Static('I have a static height', id='static'))
if __name__ == '__main__':
    HeightApp().run()