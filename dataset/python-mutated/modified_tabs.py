from textual.app import App, ComposeResult
from textual.widgets import Button, TabbedContent

class FiddleWithTabsApp(App[None]):
    CSS = '\n    TabPane:disabled {\n        background: red;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        with TabbedContent():
            yield Button()
            yield Button()
            yield Button()
            yield Button()
            yield Button()

    def on_mount(self) -> None:
        if False:
            return 10
        self.query_one(TabbedContent).disable_tab(f'tab-1')
        self.query_one(TabbedContent).disable_tab(f'tab-2')
        self.query_one(TabbedContent).hide_tab(f'tab-3')
if __name__ == '__main__':
    FiddleWithTabsApp().run()