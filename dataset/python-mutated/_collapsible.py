from __future__ import annotations
from rich.console import RenderableType
from rich.text import Text
from .. import events
from ..app import ComposeResult
from ..binding import Binding
from ..containers import Container
from ..css.query import NoMatches
from ..message import Message
from ..reactive import reactive
from ..widget import Widget
__all__ = ['Collapsible', 'CollapsibleTitle']

class CollapsibleTitle(Widget, can_focus=True):
    """Title and symbol for the Collapsible."""
    DEFAULT_CSS = '\n    CollapsibleTitle {\n        width: auto;\n        height: auto;\n        padding: 0 1 0 1;\n    }\n\n    CollapsibleTitle:hover {\n        background: $foreground 10%;\n        color: $text;\n    }\n\n    CollapsibleTitle:focus {\n        background: $accent;\n        color: $text;\n    }\n    '
    BINDINGS = [Binding('enter', 'toggle', 'Toggle collapsible', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter | Toggle the collapsible. |\n    '
    collapsed = reactive(True)

    def __init__(self, *, label: str, collapsed_symbol: str, expanded_symbol: str, collapsed: bool) -> None:
        if False:
            return 10
        super().__init__()
        self.collapsed_symbol = collapsed_symbol
        self.expanded_symbol = expanded_symbol
        self.label = label
        self.collapse = collapsed

    class Toggle(Message):
        """Request toggle."""

    async def _on_click(self, event: events.Click) -> None:
        """Inform ancestor we want to toggle."""
        event.stop()
        self.post_message(self.Toggle())

    def action_toggle(self) -> None:
        if False:
            print('Hello World!')
        'Toggle the state of the parent collapsible.'
        self.post_message(self.Toggle())

    def render(self) -> RenderableType:
        if False:
            for i in range(10):
                print('nop')
        'Compose right/down arrow and label.'
        if self.collapsed:
            return Text(f'{self.collapsed_symbol} {self.label}')
        else:
            return Text(f'{self.expanded_symbol} {self.label}')

class Collapsible(Widget):
    """A collapsible container."""
    collapsed = reactive(True)
    DEFAULT_CSS = '\n    Collapsible {\n        width: 1fr;\n        height: auto;\n        background: $boost;\n        border-top: hkey $background;\n        padding-bottom: 1;\n        padding-left: 1;\n    }\n\n    Collapsible.-collapsed > Contents {\n        display: none;\n    }\n    '

    class Contents(Container):
        DEFAULT_CSS = '\n        Contents {\n            width: 100%;\n            height: auto;\n            padding: 1 0 0 3;\n        }\n        '

    def __init__(self, *children: Widget, title: str='Toggle', collapsed: bool=True, collapsed_symbol: str='▶', expanded_symbol: str='▼', name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            return 10
        'Initialize a Collapsible widget.\n\n        Args:\n            *children: Contents that will be collapsed/expanded.\n            title: Title of the collapsed/expanded contents.\n            collapsed: Default status of the contents.\n            collapsed_symbol: Collapsed symbol before the title.\n            expanded_symbol: Expanded symbol before the title.\n            name: The name of the collapsible.\n            id: The ID of the collapsible in the DOM.\n            classes: The CSS classes of the collapsible.\n            disabled: Whether the collapsible is disabled or not.\n        '
        self._title = CollapsibleTitle(label=title, collapsed_symbol=collapsed_symbol, expanded_symbol=expanded_symbol, collapsed=collapsed)
        self._contents_list: list[Widget] = list(children)
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.collapsed = collapsed

    def on_collapsible_title_toggle(self, event: CollapsibleTitle.Toggle) -> None:
        if False:
            print('Hello World!')
        event.stop()
        self.collapsed = not self.collapsed

    def _watch_collapsed(self, collapsed: bool) -> None:
        if False:
            return 10
        'Update collapsed state when reactive is changed.'
        self._update_collapsed(collapsed)

    def _update_collapsed(self, collapsed: bool) -> None:
        if False:
            while True:
                i = 10
        'Update children to match collapsed state.'
        try:
            self._title.collapsed = collapsed
            self.set_class(collapsed, '-collapsed')
        except NoMatches:
            pass

    def _on_mount(self) -> None:
        if False:
            while True:
                i = 10
        'Initialise collapsed state.'
        self._update_collapsed(self.collapsed)

    def compose(self) -> ComposeResult:
        if False:
            print('Hello World!')
        yield self._title
        yield self.Contents(*self._contents_list)

    def compose_add_child(self, widget: Widget) -> None:
        if False:
            for i in range(10):
                print('nop')
        'When using the context manager compose syntax, we want to attach nodes to the contents.\n\n        Args:\n            widget: A Widget to add.\n        '
        self._contents_list.append(widget)