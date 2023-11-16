"""Provides a widget for switching between the display of its immediate children."""
from __future__ import annotations
from typing import Optional
from ..containers import Container
from ..css.query import NoMatches
from ..events import Mount
from ..reactive import reactive
from ..widget import Widget

class ContentSwitcher(Container):
    """A widget for switching between different children.

    Note:
        All child widgets that are to be switched between need a unique ID.
        Children that have no ID will be hidden and ignored.
    """
    DEFAULT_CSS = '\n    ContentSwitcher {\n        height: auto;\n    }\n\n    '
    current: reactive[str | None] = reactive[Optional[str]](None, init=False)
    'The ID of the currently-displayed widget.\n\n    If set to `None` then no widget is visible.\n\n    Note:\n        If set to an unknown ID, this will result in\n        [`NoMatches`][textual.css.query.NoMatches] being raised.\n    '

    def __init__(self, *children: Widget, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False, initial: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialise the content switching widget.\n\n        Args:\n            *children: The widgets to switch between.\n            name: The name of the content switcher.\n            id: The ID of the content switcher in the DOM.\n            classes: The CSS classes of the content switcher.\n            disabled: Whether the content switcher is disabled or not.\n            initial: The ID of the initial widget to show, ``None`` or empty string for the first tab.\n\n        Note:\n            If `initial` is not supplied no children will be shown to start with.\n        '
        super().__init__(*children, name=name, id=id, classes=classes, disabled=disabled)
        self._initial = initial

    def _on_mount(self, _: Mount) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Perform the initial setup of the widget once the DOM is ready.'
        initial = self._initial
        with self.app.batch_update():
            for child in self.children:
                child.display = bool(initial) and child.id == initial
        self._reactive_current = initial

    @property
    def visible_content(self) -> Widget | None:
        if False:
            print('Hello World!')
        'A reference to the currently-visible widget.\n\n        `None` if nothing is visible.\n        '
        return self.get_child_by_id(self.current) if self.current is not None else None

    def watch_current(self, old: str | None, new: str | None) -> None:
        if False:
            return 10
        'React to the current visible child choice being changed.\n\n        Args:\n            old: The old widget ID (or `None` if there was no widget).\n            new: The new widget ID (or `None` if nothing should be shown).\n        '
        with self.app.batch_update():
            if old:
                try:
                    self.get_child_by_id(old).display = False
                except NoMatches:
                    pass
            if new:
                self.get_child_by_id(new).display = True