from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.style import Style
from ..color import Color
if TYPE_CHECKING:
    from ..screen import Screen

class BackgroundScreen:
    """Tints a renderable and removes links / meta."""

    def __init__(self, screen: Screen, color: Color) -> None:
        if False:
            i = 10
            return i + 15
        'Initialize a BackgroundScreen instance.\n\n        Args:\n            screen: A Screen instance.\n            color: A color (presumably with alpha).\n        '
        self.screen = screen
        'Screen to process.'
        self.color = color
        'Color to apply (should have alpha).'

    @classmethod
    def process_segments(cls, segments: Iterable[Segment], color: Color) -> Iterable[Segment]:
        if False:
            print('Hello World!')
        'Apply tint to segments and remove meta + styles\n\n        Args:\n            segments: Incoming segments.\n            color: Color of tint.\n\n        Returns:\n            Segments with applied tint.\n        '
        from_rich_color = Color.from_rich_color
        style_from_color = Style.from_color
        _Segment = Segment
        NULL_STYLE = Style()
        for segment in segments:
            (text, style, control) = segment
            if control:
                yield segment
            else:
                style = NULL_STYLE if style is None else style.clear_meta_and_links()
                yield _Segment(text, style + style_from_color((from_rich_color(style.color) + color).rich_color if style.color is not None else None, (from_rich_color(style.bgcolor) + color).rich_color if style.bgcolor is not None else None), control)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if False:
            i = 10
            return i + 15
        segments = console.render(self.screen._compositor, options)
        color = self.color
        return self.process_segments(segments, color)