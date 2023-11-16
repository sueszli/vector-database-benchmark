"""Provides a list item widget for use with `ListView`."""
from __future__ import annotations
from textual import events
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

class ListItem(Widget, can_focus=False):
    """A widget that is an item within a `ListView`.

    A `ListItem` is designed for use within a
    [ListView][textual.widgets._list_view.ListView], please see `ListView`'s
    documentation for more details on use.
    """
    SCOPED_CSS = False
    DEFAULT_CSS = '\n    ListItem {\n        color: $text;\n        height: auto;\n        background: $panel-lighten-1;\n        overflow: hidden hidden;\n    }\n    ListItem > Widget :hover {\n        background: $boost;\n    }\n    ListView > ListItem.--highlight {\n        background: $accent 50%;\n    }\n    ListView:focus > ListItem.--highlight {\n        background: $accent;\n    }\n    ListItem > Widget {\n        height: auto;\n    }\n    '
    highlighted = reactive(False)
    'Is this item highlighted?'

    class _ChildClicked(Message):
        """For informing with the parent ListView that we were clicked"""

        def __init__(self, item: ListItem) -> None:
            if False:
                print('Hello World!')
            self.item = item
            super().__init__()

    async def _on_click(self, _: events.Click) -> None:
        self.post_message(self._ChildClicked(self))

    def watch_highlighted(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        self.set_class(value, '--highlight')