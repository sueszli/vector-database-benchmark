from collections.abc import Sequence, MutableSequence
from hscommon.gui.base import GUIObject

class Selectable(Sequence):
    """Mix-in for a ``Sequence`` that manages its selection status.

    When mixed in with a ``Sequence``, we enable it to manage its selection status. The selection
    is held as a list of ``int`` indexes. Multiple selection is supported.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._selected_indexes = []

    def _check_selection_range(self):
        if False:
            for i in range(10):
                print('nop')
        if not self:
            self._selected_indexes = []
        if not self._selected_indexes:
            return
        self._selected_indexes = [index for index in self._selected_indexes if index < len(self)]
        if not self._selected_indexes:
            self._selected_indexes = [len(self) - 1]

    def _update_selection(self):
        if False:
            i = 10
            return i + 15
        "(Virtual) Updates the model's selection appropriately.\n\n        Called after selection has been updated. Takes the table's selection and does appropriates\n        updates on the view and/or model. Common sense would dictate that when the selection doesn't\n        change, we don't update anything (and thus don't call ``_update_selection()`` at all), but\n        there are cases where it's false. For example, if our list updates its items but doesn't\n        change its selection, we probably want to update the model's selection.\n\n        By default, does nothing.\n\n        Important note: This is only called on :meth:`select`, not on changes to\n        :attr:`selected_indexes`.\n        "

    def select(self, indexes):
        if False:
            i = 10
            return i + 15
        'Update selection to ``indexes``.\n\n        :meth:`_update_selection` is called afterwards.\n\n        :param list indexes: List of ``int`` that is to become the new selection.\n        '
        if isinstance(indexes, int):
            indexes = [indexes]
        self.selected_indexes = indexes
        self._update_selection()

    @property
    def selected_index(self):
        if False:
            while True:
                i = 10
        'Points to the first selected index.\n\n        *int*. *get/set*.\n\n        Thin wrapper around :attr:`selected_indexes`. ``None`` if selection is empty. Using this\n        property only makes sense if your selectable sequence supports single selection only.\n        '
        return self._selected_indexes[0] if self._selected_indexes else None

    @selected_index.setter
    def selected_index(self, value):
        if False:
            while True:
                i = 10
        self.selected_indexes = [value]

    @property
    def selected_indexes(self):
        if False:
            i = 10
            return i + 15
        'List of selected indexes.\n\n        *list of int*. *get/set*.\n\n        When setting the value, automatically removes out-of-bounds indexes. The list is kept\n        sorted.\n        '
        return self._selected_indexes

    @selected_indexes.setter
    def selected_indexes(self, value):
        if False:
            return 10
        self._selected_indexes = value
        self._selected_indexes.sort()
        self._check_selection_range()

class SelectableList(MutableSequence, Selectable):
    """A list that can manage selection of its items.

    Subclasses :class:`Selectable`. Behaves like a ``list``.
    """

    def __init__(self, items=None):
        if False:
            return 10
        Selectable.__init__(self)
        if items:
            self._items = list(items)
        else:
            self._items = []

    def __delitem__(self, key):
        if False:
            return 10
        self._items.__delitem__(key)
        self._check_selection_range()
        self._on_change()

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        return self._items.__getitem__(key)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._items)

    def __setitem__(self, key, value):
        if False:
            return 10
        self._items.__setitem__(key, value)
        self._on_change()

    def append(self, item):
        if False:
            for i in range(10):
                print('nop')
        self._items.append(item)
        self._on_change()

    def insert(self, index, item):
        if False:
            for i in range(10):
                print('nop')
        self._items.insert(index, item)
        self._on_change()

    def remove(self, row):
        if False:
            while True:
                i = 10
        self._items.remove(row)
        self._check_selection_range()
        self._on_change()

    def _on_change(self):
        if False:
            i = 10
            return i + 15
        '(Virtual) Called whenever the contents of the list changes.\n\n        By default, does nothing.\n        '

    def search_by_prefix(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        prefix = prefix.lower()
        for (index, s) in enumerate(self):
            if s.lower().startswith(prefix):
                return index
        return -1

class GUISelectableListView:
    """Expected interface for :class:`GUISelectableList`'s view.

    *Not actually used in the code. For documentation purposes only.*

    Our view, some kind of list view or combobox, is expected to sync with the list's contents by
    appropriately behave to all callbacks in this interface.
    """

    def refresh(self):
        if False:
            for i in range(10):
                print('nop')
        'Refreshes the contents of the list widget.\n\n        Ensures that the contents of the list widget is synced with the model.\n        '

    def update_selection(self):
        if False:
            i = 10
            return i + 15
        "Update selection status.\n\n        Ensures that the list widget's selection is in sync with the model.\n        "

class GUISelectableList(SelectableList, GUIObject):
    """Cross-toolkit GUI-enabled list view.

    Represents a UI element presenting the user with a selectable list of items.

    Subclasses :class:`SelectableList` and :class:`.GUIObject`. Expected view:
    :class:`GUISelectableListView`.

    :param iterable items: If specified, items to fill the list with initially.
    """

    def __init__(self, items=None):
        if False:
            print('Hello World!')
        SelectableList.__init__(self, items)
        GUIObject.__init__(self)

    def _view_updated(self):
        if False:
            return 10
        'Refreshes the view contents with :meth:`GUISelectableListView.refresh`.\n\n        Overrides :meth:`~hscommon.gui.base.GUIObject._view_updated`.\n        '
        self.view.refresh()

    def _update_selection(self):
        if False:
            for i in range(10):
                print('nop')
        'Refreshes the view selection with :meth:`GUISelectableListView.update_selection`.\n\n        Overrides :meth:`Selectable._update_selection`.\n        '
        self.view.update_selection()

    def _on_change(self):
        if False:
            i = 10
            return i + 15
        'Refreshes the view contents with :meth:`GUISelectableListView.refresh`.\n\n        Overrides :meth:`SelectableList._on_change`.\n        '
        self.view.refresh()