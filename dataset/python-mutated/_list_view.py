from __future__ import annotations
from typing import ClassVar, Iterable, Optional
from textual.await_remove import AwaitRemove
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.events import Mount
from textual.geometry import clamp
from textual.message import Message
from textual.reactive import reactive
from textual.widget import AwaitMount, Widget
from textual.widgets._list_item import ListItem

class ListView(VerticalScroll, can_focus=True, can_focus_children=False):
    """A vertical list view widget.

    Displays a vertical list of `ListItem`s which can be highlighted and
    selected using the mouse or keyboard.

    Attributes:
        index: The index in the list that's currently highlighted.
    """
    BINDINGS: ClassVar[list[BindingType]] = [Binding('enter', 'select_cursor', 'Select', show=False), Binding('up', 'cursor_up', 'Cursor Up', show=False), Binding('down', 'cursor_down', 'Cursor Down', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter | Select the current item. |\n    | up | Move the cursor up. |\n    | down | Move the cursor down. |\n    '
    index = reactive[Optional[int]](0, always_update=True)

    class Highlighted(Message):
        """Posted when the highlighted item changes.

        Highlighted item is controlled using up/down keys.
        Can be handled using `on_list_view_highlighted` in a subclass of `ListView`
        or in a parent widget in the DOM.
        """
        ALLOW_SELECTOR_MATCH = {'item'}
        'Additional message attributes that can be used with the [`on` decorator][textual.on].'

        def __init__(self, list_view: ListView, item: ListItem | None) -> None:
            if False:
                print('Hello World!')
            super().__init__()
            self.list_view: ListView = list_view
            'The view that contains the item highlighted.'
            self.item: ListItem | None = item
            'The highlighted item, if there is one highlighted.'

        @property
        def control(self) -> ListView:
            if False:
                for i in range(10):
                    print('nop')
            'The view that contains the item highlighted.\n\n            This is an alias for [`Highlighted.list_view`][textual.widgets.ListView.Highlighted.list_view]\n            and is used by the [`on`][textual.on] decorator.\n            '
            return self.list_view

    class Selected(Message):
        """Posted when a list item is selected, e.g. when you press the enter key on it.

        Can be handled using `on_list_view_selected` in a subclass of `ListView` or in
        a parent widget in the DOM.
        """
        ALLOW_SELECTOR_MATCH = {'item'}
        'Additional message attributes that can be used with the [`on` decorator][textual.on].'

        def __init__(self, list_view: ListView, item: ListItem) -> None:
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.list_view: ListView = list_view
            'The view that contains the item selected.'
            self.item: ListItem = item
            'The selected item.'

        @property
        def control(self) -> ListView:
            if False:
                while True:
                    i = 10
            'The view that contains the item selected.\n\n            This is an alias for [`Selected.list_view`][textual.widgets.ListView.Selected.list_view]\n            and is used by the [`on`][textual.on] decorator.\n            '
            return self.list_view

    def __init__(self, *children: ListItem, initial_index: int | None=0, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Initialize a ListView.\n\n        Args:\n            *children: The ListItems to display in the list.\n            initial_index: The index that should be highlighted when the list is first mounted.\n            name: The name of the widget.\n            id: The unique ID of the widget used in CSS/query selection.\n            classes: The CSS classes of the widget.\n            disabled: Whether the ListView is disabled or not.\n        '
        super().__init__(*children, name=name, id=id, classes=classes, disabled=disabled)
        self._index = initial_index

    def _on_mount(self, _: Mount) -> None:
        if False:
            while True:
                i = 10
        'Ensure the ListView is fully-settled after mounting.'
        self.index = self._index

    @property
    def highlighted_child(self) -> ListItem | None:
        if False:
            while True:
                i = 10
        'The currently highlighted ListItem, or None if nothing is highlighted.'
        if self.index is not None and 0 <= self.index < len(self._nodes):
            list_item = self._nodes[self.index]
            assert isinstance(list_item, ListItem)
            return list_item
        else:
            return None

    def validate_index(self, index: int | None) -> int | None:
        if False:
            while True:
                i = 10
        "Clamp the index to the valid range, or set to None if there's nothing to highlight.\n\n        Args:\n            index: The index to clamp.\n\n        Returns:\n            The clamped index.\n        "
        if not self._nodes or index is None:
            return None
        return self._clamp_index(index)

    def _clamp_index(self, index: int) -> int:
        if False:
            while True:
                i = 10
        'Clamp the index to a valid value given the current list of children'
        last_index = max(len(self._nodes) - 1, 0)
        return clamp(index, 0, last_index)

    def _is_valid_index(self, index: int | None) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if the current index is valid given the current list of children'
        if index is None:
            return False
        return 0 <= index < len(self._nodes)

    def watch_index(self, old_index: int, new_index: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the highlighting when the index changes.'
        if self._is_valid_index(old_index):
            old_child = self._nodes[old_index]
            assert isinstance(old_child, ListItem)
            old_child.highlighted = False
        new_child: Widget | None
        if self._is_valid_index(new_index):
            new_child = self._nodes[new_index]
            assert isinstance(new_child, ListItem)
            new_child.highlighted = True
        else:
            new_child = None
        self._scroll_highlighted_region()
        self.post_message(self.Highlighted(self, new_child))

    def extend(self, items: Iterable[ListItem]) -> AwaitMount:
        if False:
            for i in range(10):
                print('nop')
        'Append multiple new ListItems to the end of the ListView.\n\n        Args:\n            items: The ListItems to append.\n\n        Returns:\n            An awaitable that yields control to the event loop\n                until the DOM has been updated with the new child items.\n        '
        await_mount = self.mount(*items)
        if len(self) == 1:
            self.index = 0
        return await_mount

    def append(self, item: ListItem) -> AwaitMount:
        if False:
            while True:
                i = 10
        'Append a new ListItem to the end of the ListView.\n\n        Args:\n            item: The ListItem to append.\n\n        Returns:\n            An awaitable that yields control to the event loop\n                until the DOM has been updated with the new child item.\n        '
        return self.extend([item])

    def clear(self) -> AwaitRemove:
        if False:
            for i in range(10):
                print('nop')
        'Clear all items from the ListView.\n\n        Returns:\n            An awaitable that yields control to the event loop until\n                the DOM has been updated to reflect all children being removed.\n        '
        await_remove = self.query('ListView > ListItem').remove()
        self.index = None
        return await_remove

    def action_select_cursor(self) -> None:
        if False:
            while True:
                i = 10
        'Select the current item in the list.'
        selected_child = self.highlighted_child
        if selected_child is None:
            return
        self.post_message(self.Selected(self, selected_child))

    def action_cursor_down(self) -> None:
        if False:
            print('Hello World!')
        'Highlight the next item in the list.'
        if self.index is None:
            self.index = 0
            return
        self.index += 1

    def action_cursor_up(self) -> None:
        if False:
            return 10
        'Highlight the previous item in the list.'
        if self.index is None:
            self.index = 0
            return
        self.index -= 1

    def _on_list_item__child_clicked(self, event: ListItem._ChildClicked) -> None:
        if False:
            print('Hello World!')
        self.focus()
        self.index = self._nodes.index(event.item)
        self.post_message(self.Selected(self, event.item))

    def _scroll_highlighted_region(self) -> None:
        if False:
            print('Hello World!')
        'Used to keep the highlighted index within vision'
        if self.highlighted_child is not None:
            self.scroll_to_widget(self.highlighted_child, animate=False)

    def __len__(self):
        if False:
            return 10
        return len(self._nodes)