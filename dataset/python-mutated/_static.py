from __future__ import annotations
from rich.console import RenderableType
from rich.protocol import is_renderable
from rich.text import Text
from ..errors import RenderError
from ..widget import Widget

def _check_renderable(renderable: object):
    if False:
        i = 10
        return i + 15
    'Check if a renderable conforms to the Rich Console protocol\n    (https://rich.readthedocs.io/en/latest/protocol.html)\n\n    Args:\n        renderable: A potentially renderable object.\n\n    Raises:\n        RenderError: If the object can not be rendered.\n    '
    if not is_renderable(renderable):
        raise RenderError(f'unable to render {renderable!r}; a string, Text, or other Rich renderable is required')

class Static(Widget, inherit_bindings=False):
    """A widget to display simple static content, or use as a base class for more complex widgets.

    Args:
        renderable: A Rich renderable, or string containing console markup.
        expand: Expand content if required to fill container.
        shrink: Shrink content if required to fill container.
        markup: True if markup should be parsed and rendered.
        name: Name of widget.
        id: ID of Widget.
        classes: Space separated list of class names.
        disabled: Whether the static is disabled or not.
    """
    DEFAULT_CSS = '\n    Static {\n        height: auto;\n    }\n    '
    _renderable: RenderableType

    def __init__(self, renderable: RenderableType='', *, expand: bool=False, shrink: bool=False, markup: bool=True, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.expand = expand
        self.shrink = shrink
        self.markup = markup
        self.renderable = renderable
        _check_renderable(renderable)

    @property
    def renderable(self) -> RenderableType:
        if False:
            while True:
                i = 10
        return self._renderable or ''

    @renderable.setter
    def renderable(self, renderable: RenderableType) -> None:
        if False:
            return 10
        if isinstance(renderable, str):
            if self.markup:
                self._renderable = Text.from_markup(renderable)
            else:
                self._renderable = Text(renderable)
        else:
            self._renderable = renderable

    def render(self) -> RenderableType:
        if False:
            for i in range(10):
                print('nop')
        "Get a rich renderable for the widget's content.\n\n        Returns:\n            A rich renderable.\n        "
        return self._renderable

    def update(self, renderable: RenderableType='') -> None:
        if False:
            return 10
        "Update the widget's content area with new text or Rich renderable.\n\n        Args:\n            renderable: A new rich renderable. Defaults to empty renderable;\n        "
        _check_renderable(renderable)
        self.renderable = renderable
        self.refresh(layout=True)