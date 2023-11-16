"""Provides a pretty-printing widget."""
from __future__ import annotations
from typing import Any
from rich.pretty import Pretty as PrettyRenderable
from ..widget import Widget

class Pretty(Widget):
    """A pretty-printing widget.

    Used to pretty-print any object.
    """
    DEFAULT_CSS = '\n    Pretty {\n        height: auto;\n    }\n    '

    def __init__(self, object: Any, *, name: str | None=None, id: str | None=None, classes: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Initialise the `Pretty` widget.\n\n        Args:\n            object: The object to pretty-print.\n            name: The name of the pretty widget.\n            id: The ID of the pretty in the DOM.\n            classes: The CSS classes of the pretty.\n        '
        super().__init__(name=name, id=id, classes=classes)
        self._renderable = PrettyRenderable(object)

    def render(self) -> PrettyRenderable:
        if False:
            for i in range(10):
                print('nop')
        'Render the pretty-printed object.\n\n        Returns:\n            The rendered pretty-print.\n        '
        return self._renderable

    def update(self, object: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the content of the pretty widget.\n\n        Args:\n            object: The object to pretty-print.\n        '
        self._renderable = PrettyRenderable(object)
        self.refresh(layout=True)