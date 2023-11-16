from __future__ import annotations
from typing import Callable, ClassVar, Optional, Sequence
from ..app import RenderResult
from ..reactive import reactive
from ..renderables.sparkline import Sparkline as SparklineRenderable
from ..widget import Widget

def _max_factory() -> Callable[[Sequence[float]], float]:
    if False:
        return 10
    'Callable that returns the built-in max to initialise a reactive.'
    return max

class Sparkline(Widget):
    """A sparkline widget to display numerical data."""
    COMPONENT_CLASSES: ClassVar[set[str]] = {'sparkline--max-color', 'sparkline--min-color'}
    '\n    Use these component classes to define the two colors that the sparkline\n    interpolates to represent its numerical data.\n\n    Note:\n        These two component classes are used exclusively for the _color_ of the\n        sparkline widget. Setting any style other than [`color`](/styles/color.md)\n        will have no effect.\n\n    | Class | Description |\n    | :- | :- |\n    | `sparkline--max-color` | The color used for the larger values in the data. |\n    | `sparkline--min-color` | The colour used for the smaller values in the data. |\n    '
    DEFAULT_CSS = '\n    Sparkline {\n        height: 1;\n    }\n    Sparkline > .sparkline--max-color {\n        color: $accent;\n    }\n    Sparkline > .sparkline--min-color {\n        color: $accent 30%;\n    }\n    '
    data = reactive[Optional[Sequence[float]]](None)
    'The data that populates the sparkline.'
    summary_function = reactive[Callable[[Sequence[float]], float]](_max_factory)
    'The function that computes the value that represents each bar.'

    def __init__(self, data: Sequence[float] | None=None, *, summary_function: Callable[[Sequence[float]], float] | None=None, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            print('Hello World!')
        'Initialize a sparkline widget.\n\n        Args:\n            data: The initial data to populate the sparkline with.\n            summary_function: Summarises bar values into a single value used to\n                represent each bar.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes for the widget.\n            disabled: Whether the widget is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.data = data
        if summary_function is not None:
            self.summary_function = summary_function

    def render(self) -> RenderResult:
        if False:
            for i in range(10):
                print('nop')
        'Renders the sparkline when there is data available.'
        if not self.data:
            return '<empty sparkline>'
        (_, base) = self.background_colors
        return SparklineRenderable(self.data, width=self.size.width, min_color=(base + self.get_component_styles('sparkline--min-color').color).rich_color, max_color=(base + self.get_component_styles('sparkline--max-color').color).rich_color, summary_function=self.summary_function)