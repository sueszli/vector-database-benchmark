from rich.text import Text
from textual.app import App, ComposeResult, RenderResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer
from textual.widget import Widget

class Tester(Widget, can_focus=True):
    COMPONENT_CLASSES = {'tester--text'}
    DEFAULT_CSS = '\n    Tester {\n        height: auto;\n    }\n    \n    Tester:focus > .tester--text {\n        background: red;\n    }\n    '

    def __init__(self, n: int) -> None:
        if False:
            i = 10
            return i + 15
        self.n = n
        super().__init__()

    def render(self) -> RenderResult:
        if False:
            print('Hello World!')
        return Text(f'test widget {self.n}', style=self.get_component_rich_style('tester--text'))

class StyleBugApp(App[None]):

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield Header()
        with VerticalScroll():
            for n in range(40):
                yield Tester(n)
        yield Footer()
if __name__ == '__main__':
    StyleBugApp().run()