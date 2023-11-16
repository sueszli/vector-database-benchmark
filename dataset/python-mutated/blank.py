from __future__ import annotations
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
from ..color import Color

class Blank:
    """Draw solid background color."""

    def __init__(self, color: Color | str='transparent') -> None:
        if False:
            for i in range(10):
                print('nop')
        background = Color.parse(color)
        self._style = Style.from_color(bgcolor=background.rich_color)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if False:
            while True:
                i = 10
        width = options.max_width
        height = options.height or options.max_height
        segment = Segment(' ' * width, self._style)
        line = Segment.line()
        for _ in range(height):
            yield segment
            yield line