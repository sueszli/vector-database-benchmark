from collections import namedtuple
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QAction
from hscommon.trans import trget
from hscommon.util import dedupe
tr = trget('ui')
MenuEntry = namedtuple('MenuEntry', 'menu fixedItemCount')

class Recent(QObject):

    def __init__(self, app, pref_name, max_item_count=10, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self._app = app
        self._menuEntries = []
        self._prefName = pref_name
        self._maxItemCount = max_item_count
        self._items = []
        self._loadFromPrefs()
        self._app.willSavePrefs.connect(self._saveToPrefs)

    def _loadFromPrefs(self):
        if False:
            while True:
                i = 10
        items = getattr(self._app.prefs, self._prefName)
        if not isinstance(items, list):
            items = []
        self._items = items

    def _insertItem(self, item):
        if False:
            return 10
        self._items = dedupe([item] + self._items)[:self._maxItemCount]

    def _refreshMenu(self, menu_entry):
        if False:
            i = 10
            return i + 15
        (menu, fixed_item_count) = menu_entry
        for action in menu.actions()[fixed_item_count:]:
            menu.removeAction(action)
        for item in self._items:
            action = QAction(item, menu)
            action.setData(item)
            action.triggered.connect(self.menuItemWasClicked)
            menu.addAction(action)
        menu.addSeparator()
        action = QAction(tr('Clear List'), menu)
        action.triggered.connect(self.clear)
        menu.addAction(action)

    def _refreshAllMenus(self):
        if False:
            return 10
        for menu_entry in self._menuEntries:
            self._refreshMenu(menu_entry)

    def _saveToPrefs(self):
        if False:
            for i in range(10):
                print('nop')
        setattr(self._app.prefs, self._prefName, self._items)

    def addMenu(self, menu):
        if False:
            while True:
                i = 10
        menu_entry = MenuEntry(menu, len(menu.actions()))
        self._menuEntries.append(menu_entry)
        self._refreshMenu(menu_entry)

    def clear(self):
        if False:
            print('Hello World!')
        self._items = []
        self._refreshAllMenus()
        self.itemsChanged.emit()

    def insertItem(self, item):
        if False:
            return 10
        self._insertItem(str(item))
        self._refreshAllMenus()
        self.itemsChanged.emit()

    def isEmpty(self):
        if False:
            while True:
                i = 10
        return not bool(self._items)

    def menuItemWasClicked(self):
        if False:
            for i in range(10):
                print('nop')
        action = self.sender()
        if action is not None:
            item = action.data()
            self.mustOpenItem.emit(item)
            self._refreshAllMenus()
    mustOpenItem = pyqtSignal(str)
    itemsChanged = pyqtSignal()