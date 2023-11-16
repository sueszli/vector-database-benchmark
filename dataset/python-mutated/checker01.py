from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.strip import Strip
from textual.widget import Widget

class CheckerBoard(Widget):
    """Render an 8x8 checkerboard."""

    def render_line(self, y: int) -> Strip:
        if False:
            return 10
        'Render a line of the widget. y is relative to the top of the widget.'
        row_index = y // 4
        if row_index >= 8:
            return Strip.blank(self.size.width)
        is_odd = row_index % 2
        white = Style.parse('on white')
        black = Style.parse('on black')
        segments = [Segment(' ' * 8, black if (column + is_odd) % 2 else white) for column in range(8)]
        strip = Strip(segments, 8 * 8)
        return strip

class BoardApp(App):
    """A simple app to show our widget."""

    def compose(self) -> ComposeResult:
        if False:
            for i in range(10):
                print('nop')
        yield CheckerBoard()
if __name__ == '__main__':
    app = BoardApp()
    app.run()