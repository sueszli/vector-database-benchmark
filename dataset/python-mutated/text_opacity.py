import functools
from typing import Iterable, Tuple, cast
from rich.cells import cell_len
from rich.color import Color
from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.segment import Segment
from rich.style import Style
from textual.renderables._blend_colors import blend_colors

@functools.lru_cache(maxsize=1024)
def _get_blended_style_cached(bg_color: Color, fg_color: Color, opacity: float) -> Style:
    if False:
        while True:
            i = 10
    'Blend from one color to another.\n\n    Cached because when a UI is static the opacity will be constant.\n\n    Args:\n        bg_color: Background color.\n        fg_color: Foreground color.\n        opacity: Opacity.\n\n    Returns:\n        Resulting style.\n    '
    return Style.from_color(color=blend_colors(bg_color, fg_color, ratio=opacity), bgcolor=bg_color)

class TextOpacity:
    """Blend foreground in to background."""

    def __init__(self, renderable: RenderableType, opacity: float=1.0) -> None:
        if False:
            while True:
                i = 10
        'Wrap a renderable to blend foreground color into the background color.\n\n        Args:\n            renderable: The RenderableType to manipulate.\n            opacity: The opacity as a float. A value of 1.0 means text is fully visible.\n        '
        self.renderable = renderable
        self.opacity = opacity

    @classmethod
    def process_segments(cls, segments: Iterable[Segment], opacity: float) -> Iterable[Segment]:
        if False:
            for i in range(10):
                print('nop')
        'Apply opacity to segments.\n\n        Args:\n            segments: Incoming segments.\n            opacity: Opacity to apply.\n\n        Returns:\n            Segments with applied opacity.\n        '
        _Segment = Segment
        _from_color = Style.from_color
        if opacity == 0:
            for (text, style, _control) in cast(Iterable[Tuple[str, Style, object]], segments):
                invisible_style = _from_color(bgcolor=style.bgcolor)
                yield _Segment(cell_len(text) * ' ', invisible_style)
        else:
            for segment in segments:
                (text, style, control) = cast(Tuple[str, Style, object], segment)
                if not style:
                    yield segment
                    continue
                color = style.color
                bgcolor = style.bgcolor
                if color and color.triplet and bgcolor and bgcolor.triplet:
                    color_style = _get_blended_style_cached(bgcolor, color, opacity)
                    yield _Segment(text, style + color_style)
                else:
                    yield segment

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        if False:
            for i in range(10):
                print('nop')
        segments = console.render(self.renderable, options)
        return self.process_segments(segments, self.opacity)