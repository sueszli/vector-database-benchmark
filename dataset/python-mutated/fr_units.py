from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Static

class StaticText(Static):
    pass

class FRApp(App):
    CSS = '\n    StaticText {\n        height: 1fr;\n        background: $boost;\n        border: heavy white;\n    }\n    #foo {\n        width: 10;\n    }\n    #bar {\n        width: 1fr;\n    }\n    #baz {\n        width: 8;\n    }\n    #header {\n        height: 1fr\n    }\n\n    Horizontal {\n        height: 2fr;\n    }\n\n    #footer {\n        height: 4;\n    }\n    \n    '

    def compose(self) -> ComposeResult:
        if False:
            return 10
        yield VerticalScroll(StaticText('HEADER', id='header'), Horizontal(StaticText('foo', id='foo'), StaticText('bar', id='bar'), StaticText('baz', id='baz')), StaticText('FOOTER', id='footer'))
if __name__ == '__main__':
    app = FRApp()
    app.run()