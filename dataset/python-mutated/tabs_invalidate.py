from textual.app import App, ComposeResult
from textual.widgets import Label, TabbedContent, TabPane

class TabApp(App):
    CSS = '\n    TabPane {\n    border: solid blue;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        with TabbedContent():
            with TabPane('Tab 1', id='tab-1'):
                yield Label('hello')
                yield Label('hello')
                yield Label('hello')
            with TabPane('Tab 2', id='tab-2'):
                yield Label('world')
if __name__ == '__main__':
    app = TabApp()
    app.run()