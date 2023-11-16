from __future__ import annotations
from time import time
from typing import Awaitable
from rich.console import RenderableType
from rich.style import Style
from rich.text import Text
from ..color import Gradient
from ..css.query import NoMatches
from ..events import Mount
from ..widget import AwaitMount, Widget

class LoadingIndicator(Widget):
    """Display an animated loading indicator."""
    DEFAULT_CSS = '\n    LoadingIndicator {\n        width: 100%;\n        height: 100%;\n        min-height: 1;\n        content-align: center middle;\n        color: $accent;\n    }\n    LoadingIndicator.-overlay {\n        overlay: screen;\n        background: $boost;\n    }\n    '

    def __init__(self, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False):
        if False:
            return 10
        'Initialize a loading indicator.\n\n        Args:\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes for the widget.\n            disabled: Whether the widget is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._start_time: float = 0.0
        'The time the loading indicator was mounted (a Unix timestamp).'

    def apply(self, widget: Widget) -> AwaitMount:
        if False:
            return 10
        'Apply the loading indicator to a `widget`.\n\n        This will overlay the given widget with a loading indicator.\n\n        Args:\n            widget: A widget.\n\n        Returns:\n            AwaitMount: An awaitable for mounting the indicator.\n        '
        self.add_class('-overlay')
        await_mount = widget.mount(self, before=0)
        return await_mount

    @classmethod
    def clear(cls, widget: Widget) -> Awaitable:
        if False:
            i = 10
            return i + 15
        'Clear any loading indicator from the given widget.\n\n        Args:\n            widget: Widget to clear the loading indicator from.\n\n        Returns:\n            Optional awaitable.\n        '
        try:
            await_remove = widget.get_child_by_type(cls).remove()
        except NoMatches:

            async def null() -> None:
                """Nothing to remove"""
                return None
            return null()
        return await_remove

    def _on_mount(self, _: Mount) -> None:
        if False:
            while True:
                i = 10
        self._start_time = time()
        self.auto_refresh = 1 / 16

    def render(self) -> RenderableType:
        if False:
            while True:
                i = 10
        elapsed = time() - self._start_time
        speed = 0.8
        dot = '●'
        (_, _, background, color) = self.colors
        gradient = Gradient((0.0, background.blend(color, 0.1)), (0.7, color), (1.0, color.lighten(0.1)))
        blends = [(elapsed * speed - dot_number / 8) % 1 for dot_number in range(5)]
        dots = [(f'{dot} ', Style.from_color(gradient.get_color((1 - blend) ** 2).rich_color)) for blend in blends]
        indicator = Text.assemble(*dots)
        indicator.rstrip()
        return indicator