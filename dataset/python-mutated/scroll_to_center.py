from textual.app import App, ComposeResult
from textual.containers import HorizontalScroll, VerticalScroll
from textual.widgets import Label

class MyApp(App[None]):
    AUTO_FOCUS = ''
    CSS = '\n    VerticalScroll {\n        border: round $primary;\n    }\n    #vertical {\n        height: 21;\n    }\n    HorizontalScroll {\n        border: round $secondary;\n        height: auto;\n    }\n    Label {\n        height: auto;\n        width: auto;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            i = 10
            return i + 15
        yield Label(('SPAM\n' * 53)[:-1])
        with VerticalScroll(id='vertical'):
            yield Label(('SPAM\n' * 78)[:-1])
            with HorizontalScroll():
                yield Label(('v\n' * 17)[:-1])
                yield Label('@' * 302)
                yield Label('[red]>>bullseye<<[/red]', id='bullseye')
                yield Label('@' * 99)
            yield Label(('SPAM\n' * 49)[:-1])
        yield Label(('SPAM\n' * 51)[:-1])

    def key_s(self) -> None:
        if False:
            return 10
        self.screen.scroll_to_center(self.query_one('#bullseye'))
if __name__ == '__main__':
    MyApp().run()