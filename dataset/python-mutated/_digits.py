from __future__ import annotations
from typing import cast
from rich.align import Align, AlignMethod
from rich.console import RenderableType
from ..geometry import Size
from ..renderables.digits import Digits as DigitsRenderable
from ..widget import Widget

class Digits(Widget):
    """A widget to display numerical values using a 3x3 grid of unicode characters."""
    DEFAULT_CSS = '\n    Digits {\n        width: 1fr;\n        height: auto;\n        text-align: left;\n        text-style: bold;\n        box-sizing: border-box;\n    }\n    '

    def __init__(self, value: str='', *, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Args:\n            value: Value to display in widget.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes of the widget.\n            disabled: Whether the widget is disabled or not.\n\n        '
        if not isinstance(value, str):
            raise TypeError('value must be a str')
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._value = value

    @property
    def value(self) -> str:
        if False:
            print('Hello World!')
        'The current value displayed in the Digits.'
        return self._value

    def update(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Update the Digits with a new value.\n\n        Args:\n            value: New value to display.\n\n        Raises:\n            ValueError: If the value isn't a `str`.\n        "
        if not isinstance(value, str):
            raise TypeError('value must be a str')
        layout_required = len(value) != len(self._value) or DigitsRenderable.get_width(self._value) != DigitsRenderable.get_width(value)
        self._value = value
        self.refresh(layout=layout_required)

    def render(self) -> RenderableType:
        if False:
            return 10
        'Render digits.'
        rich_style = self.rich_style
        digits = DigitsRenderable(self._value, rich_style)
        text_align = self.styles.text_align
        align = 'left' if text_align not in {'left', 'center', 'right'} else text_align
        return Align(digits, cast(AlignMethod, align), rich_style)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        if False:
            return 10
        'Called by textual to get the width of the content area.\n\n        Args:\n            container: Size of the container (immediate parent) widget.\n            viewport: Size of the viewport.\n\n        Returns:\n            The optimal width of the content.\n        '
        return DigitsRenderable.get_width(self._value)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        if False:
            return 10
        'Called by Textual to get the height of the content area.\n\n        Args:\n            container: Size of the container (immediate parent) widget.\n            viewport: Size of the viewport.\n            width: Width of renderable.\n\n        Returns:\n            The height of the content.\n        '
        return 3