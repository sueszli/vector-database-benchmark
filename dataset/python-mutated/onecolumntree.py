from qtpy import PYQT5
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QTreeWidget
from spyder.api.widgets.mixins import SpyderWidgetMixin
from spyder.config.base import _
from spyder.utils.icon_manager import ima
from spyder.utils.qthelpers import get_item_user_text

class OneColumnTreeActions:
    CollapseAllAction = 'collapse_all_action'
    ExpandAllAction = 'expand_all_action'
    RestoreAction = 'restore_action'
    CollapseSelectionAction = 'collapse_selection_action'
    ExpandSelectionAction = 'expand_selection_action'

class OneColumnTreeContextMenuSections:
    Global = 'global_section'
    Restore = 'restore_section'
    Section = 'section_section'
    History = 'history_section'

class OneColumnTree(QTreeWidget, SpyderWidgetMixin):
    """
    One-column tree widget with context menu.
    """

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        if PYQT5:
            super().__init__(parent, class_parent=parent)
        else:
            QTreeWidget.__init__(self, parent)
            SpyderWidgetMixin.__init__(self, class_parent=parent)
        self.__expanded_state = None
        self.setItemsExpandable(True)
        self.setColumnCount(1)
        self.collapse_all_action = None
        self.collapse_selection_action = None
        self.expand_all_action = None
        self.expand_selection_action = None
        self.setup()
        self.common_actions = self.setup_common_actions()
        self.itemActivated.connect(self.activated)
        self.itemClicked.connect(self.clicked)
        self.itemSelectionChanged.connect(self.item_selection_changed)
        self.setMouseTracking(True)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.header().setStretchLastSection(False)
        self.item_selection_changed()

    def setup(self):
        if False:
            return 10
        self.menu = self.create_menu('context_menu')
        self.collapse_all_action = self.create_action(OneColumnTreeActions.CollapseAllAction, text=_('Collapse all'), icon=ima.icon('collapse'), triggered=self.collapseAll, register_shortcut=False)
        self.expand_all_action = self.create_action(OneColumnTreeActions.ExpandAllAction, text=_('Expand all'), icon=ima.icon('expand'), triggered=self.expandAll, register_shortcut=False)
        self.restore_action = self.create_action(OneColumnTreeActions.RestoreAction, text=_('Restore'), tip=_('Restore original tree layout'), icon=ima.icon('restore'), triggered=self.restore, register_shortcut=False)
        self.collapse_selection_action = self.create_action(OneColumnTreeActions.CollapseSelectionAction, text=_('Collapse section'), icon=ima.icon('collapse_selection'), triggered=self.collapse_selection, register_shortcut=False)
        self.expand_selection_action = self.create_action(OneColumnTreeActions.ExpandSelectionAction, text=_('Expand section'), icon=ima.icon('expand_selection'), triggered=self.expand_selection, register_shortcut=False)
        for item in [self.collapse_all_action, self.expand_all_action]:
            self.add_item_to_menu(item, self.menu, section=OneColumnTreeContextMenuSections.Global)
        self.add_item_to_menu(self.restore_action, self.menu, section=OneColumnTreeContextMenuSections.Restore)
        for item in [self.collapse_selection_action, self.expand_selection_action]:
            self.add_item_to_menu(item, self.menu, section=OneColumnTreeContextMenuSections.Section)

    def update_actions(self):
        if False:
            print('Hello World!')
        pass

    def activated(self, item):
        if False:
            while True:
                i = 10
        'Double-click event'
        raise NotImplementedError

    def clicked(self, item):
        if False:
            return 10
        pass

    def set_title(self, title):
        if False:
            for i in range(10):
                print('nop')
        self.setHeaderLabels([title])

    def setup_common_actions(self):
        if False:
            return 10
        'Setup context menu common actions'
        return [self.collapse_all_action, self.expand_all_action, self.collapse_selection_action, self.expand_selection_action]

    def get_menu_actions(self):
        if False:
            print('Hello World!')
        'Returns a list of menu actions'
        items = self.selectedItems()
        actions = self.get_actions_from_items(items)
        if actions:
            actions.append(None)
        actions += self.common_actions
        return actions

    def get_actions_from_items(self, items):
        if False:
            print('Hello World!')
        return []

    @Slot()
    def restore(self):
        if False:
            i = 10
            return i + 15
        self.collapseAll()
        for item in self.get_top_level_items():
            self.expandItem(item)

    def is_item_expandable(self, item):
        if False:
            i = 10
            return i + 15
        'To be reimplemented in child class\n        See example in project explorer widget'
        return True

    def __expand_item(self, item):
        if False:
            return 10
        if self.is_item_expandable(item):
            self.expandItem(item)
            for index in range(item.childCount()):
                child = item.child(index)
                self.__expand_item(child)

    @Slot()
    def expand_selection(self):
        if False:
            return 10
        items = self.selectedItems()
        if not items:
            items = self.get_top_level_items()
        for item in items:
            self.__expand_item(item)
        if items:
            self.scrollToItem(items[0])

    def __collapse_item(self, item):
        if False:
            return 10
        self.collapseItem(item)
        for index in range(item.childCount()):
            child = item.child(index)
            self.__collapse_item(child)

    @Slot()
    def collapse_selection(self):
        if False:
            print('Hello World!')
        items = self.selectedItems()
        if not items:
            items = self.get_top_level_items()
        for item in items:
            self.__collapse_item(item)
        if items:
            self.scrollToItem(items[0])

    def item_selection_changed(self):
        if False:
            return 10
        'Item selection has changed'
        is_selection = len(self.selectedItems()) > 0
        self.expand_selection_action.setEnabled(is_selection)
        self.collapse_selection_action.setEnabled(is_selection)

    def get_top_level_items(self):
        if False:
            print('Hello World!')
        'Iterate over top level items'
        return [self.topLevelItem(_i) for _i in range(self.topLevelItemCount())]

    def get_items(self):
        if False:
            i = 10
            return i + 15
        'Return items (excluding top level items)'
        itemlist = []

        def add_to_itemlist(item):
            if False:
                for i in range(10):
                    print('nop')
            for index in range(item.childCount()):
                citem = item.child(index)
                itemlist.append(citem)
                add_to_itemlist(citem)
        for tlitem in self.get_top_level_items():
            add_to_itemlist(tlitem)
        return itemlist

    def get_scrollbar_position(self):
        if False:
            i = 10
            return i + 15
        return (self.horizontalScrollBar().value(), self.verticalScrollBar().value())

    def set_scrollbar_position(self, position):
        if False:
            return 10
        (hor, ver) = position
        self.horizontalScrollBar().setValue(hor)
        self.verticalScrollBar().setValue(ver)

    def get_expanded_state(self):
        if False:
            print('Hello World!')
        self.save_expanded_state()
        return self.__expanded_state

    def set_expanded_state(self, state):
        if False:
            return 10
        self.__expanded_state = state
        self.restore_expanded_state()

    def save_expanded_state(self):
        if False:
            return 10
        'Save all items expanded state'
        self.__expanded_state = {}

        def add_to_state(item):
            if False:
                print('Hello World!')
            user_text = get_item_user_text(item)
            self.__expanded_state[hash(user_text)] = item.isExpanded()

        def browse_children(item):
            if False:
                return 10
            add_to_state(item)
            for index in range(item.childCount()):
                citem = item.child(index)
                user_text = get_item_user_text(citem)
                self.__expanded_state[hash(user_text)] = citem.isExpanded()
                browse_children(citem)
        for tlitem in self.get_top_level_items():
            browse_children(tlitem)

    def restore_expanded_state(self):
        if False:
            return 10
        'Restore all items expanded state'
        if self.__expanded_state is None:
            return
        for item in self.get_items() + self.get_top_level_items():
            user_text = get_item_user_text(item)
            is_expanded = self.__expanded_state.get(hash(user_text))
            if is_expanded is not None:
                item.setExpanded(is_expanded)

    def sort_top_level_items(self, key):
        if False:
            for i in range(10):
                print('nop')
        'Sorting tree wrt top level items'
        self.save_expanded_state()
        items = sorted([self.takeTopLevelItem(0) for index in range(self.topLevelItemCount())], key=key)
        for (index, item) in enumerate(items):
            self.insertTopLevelItem(index, item)
        self.restore_expanded_state()

    def contextMenuEvent(self, event):
        if False:
            return 10
        'Override Qt method'
        self.menu.popup(event.globalPos())

    def mouseMoveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Change cursor shape.'
        index = self.indexAt(event.pos())
        if index.isValid():
            vrect = self.visualRect(index)
            item_identation = vrect.x() - self.visualRect(self.rootIndex()).x()
            if event.pos().x() > item_identation:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)