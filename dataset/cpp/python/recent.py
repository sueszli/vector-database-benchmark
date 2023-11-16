# Created By: Virgil Dupras
# Created On: 2009-11-12
# Copyright 2015 Hardcoded Software (http://www.hardcoded.net)
#
# This software is licensed under the "GPLv3" License as described in the "LICENSE" file,
# which should be included with this package. The terms are also available at
# http://www.gnu.org/licenses/gpl-3.0.html

from collections import namedtuple

from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QAction

from hscommon.trans import trget
from hscommon.util import dedupe

tr = trget("ui")

MenuEntry = namedtuple("MenuEntry", "menu fixedItemCount")


class Recent(QObject):
    def __init__(self, app, pref_name, max_item_count=10, **kwargs):
        super().__init__(**kwargs)
        self._app = app
        self._menuEntries = []
        self._prefName = pref_name
        self._maxItemCount = max_item_count
        self._items = []
        self._loadFromPrefs()

        self._app.willSavePrefs.connect(self._saveToPrefs)

    # --- Private
    def _loadFromPrefs(self):
        items = getattr(self._app.prefs, self._prefName)
        if not isinstance(items, list):
            items = []
        self._items = items

    def _insertItem(self, item):
        self._items = dedupe([item] + self._items)[: self._maxItemCount]

    def _refreshMenu(self, menu_entry):
        menu, fixed_item_count = menu_entry
        for action in menu.actions()[fixed_item_count:]:
            menu.removeAction(action)
        for item in self._items:
            action = QAction(item, menu)
            action.setData(item)
            action.triggered.connect(self.menuItemWasClicked)
            menu.addAction(action)
        menu.addSeparator()
        action = QAction(tr("Clear List"), menu)
        action.triggered.connect(self.clear)
        menu.addAction(action)

    def _refreshAllMenus(self):
        for menu_entry in self._menuEntries:
            self._refreshMenu(menu_entry)

    def _saveToPrefs(self):
        setattr(self._app.prefs, self._prefName, self._items)

    # --- Public
    def addMenu(self, menu):
        menu_entry = MenuEntry(menu, len(menu.actions()))
        self._menuEntries.append(menu_entry)
        self._refreshMenu(menu_entry)

    def clear(self):
        self._items = []
        self._refreshAllMenus()
        self.itemsChanged.emit()

    def insertItem(self, item):
        self._insertItem(str(item))
        self._refreshAllMenus()
        self.itemsChanged.emit()

    def isEmpty(self):
        return not bool(self._items)

    # --- Event Handlers
    def menuItemWasClicked(self):
        action = self.sender()
        if action is not None:
            item = action.data()
            self.mustOpenItem.emit(item)
            self._refreshAllMenus()

    # --- Signals
    mustOpenItem = pyqtSignal(str)
    itemsChanged = pyqtSignal()
