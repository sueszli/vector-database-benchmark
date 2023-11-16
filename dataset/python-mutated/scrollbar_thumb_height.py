from rich.segment import Segment
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer
from textual.scroll_view import ScrollView
from textual.strip import Strip
from textual.geometry import Size

class TestScrollView(ScrollView, can_focus=True):

    def __init__(self, height: int, border_title: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.virtual_size = Size(0, height)
        self.border_title = border_title

    def render_line(self, y: int) -> Strip:
        if False:
            i = 10
            return i + 15
        return Strip([Segment(f'Welcome to line {self.scroll_offset.y + y}')])

class ScrollViewTester(App[None]):
    """Check the scrollbar fits the end."""
    CSS = '\n    TestScrollView {\n        background: $primary-darken-2;\n        border: round red;\n    }\n    '

    def compose(self) -> ComposeResult:
        if False:
            while True:
                i = 10
        yield Header()
        yield TestScrollView(height=1000, border_title=f'1')
        yield Footer()

    def on_ready(self) -> None:
        if False:
            return 10
        self.query_one(TestScrollView).scroll_end(animate=False)
if __name__ == '__main__':
    ScrollViewTester().run()