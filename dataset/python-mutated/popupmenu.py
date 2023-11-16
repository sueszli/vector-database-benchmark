from typing import List, Union
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMenu, QAction, QWidget
import autokey.configmanager.configmanager as cm
import autokey.configmanager.configmanager_constants as cm_constants
import autokey.model
import autokey.model.abstract_hotkey
import autokey.model.folder
import autokey.model.phrase
import autokey.model.script
import autokey.service
logger = __import__('autokey.logger').logger.get_logger(__name__)
FolderList = List[autokey.model.folder.Folder]
Item = Union[autokey.model.script.Script, autokey.model.phrase.Phrase]

class PopupMenu(QMenu):

    def __init__(self, service: autokey.service.Service, folders: FolderList=None, items: List[Item]=None, on_desktop: bool=True, title: str=None, parent=None):
        if False:
            return 10
        super(PopupMenu, self).__init__(parent)
        if items is None:
            items = []
        if folders is None:
            folders = []
        self.setFocusPolicy(Qt.StrongFocus)
        self.service = service
        self._on_desktop = on_desktop
        if title is not None:
            self.setTitle(title)
        if cm.ConfigManager.SETTINGS[cm_constants.SORT_BY_USAGE_COUNT]:
            logger.debug('Sorting phrase menu by usage count')
            folders.sort(key=lambda obj: obj.usageCount, reverse=True)
            items.sort(key=lambda obj: obj.usageCount, reverse=True)
        else:
            logger.debug('Sorting phrase menu by item name/title')
            folders.sort(key=lambda obj: str(obj))
            items.sort(key=lambda obj: str(obj))
        if len(folders) == 1 and len(items) == 0 and on_desktop:
            self.setTitle(folders[0].title)
            for folder in folders[0].folders:
                sub_menu_item = SubMenu(self._getMnemonic(folder.title), self, service, folder.folders, folder.items, False)
                self.addAction(sub_menu_item)
            if folders[0].folders:
                self.addSeparator()
            self._add_items_to_self(folders[0].items, on_desktop)
        else:
            for folder in folders:
                sub_menu_item = SubMenu(self._getMnemonic(folder.title), self, service, folder.folders, folder.items, False)
                self.addAction(sub_menu_item)
            if folders:
                self.addSeparator()
            self._add_items_to_self(items, on_desktop)

    def _add_item(self, description, item):
        if False:
            for i in range(10):
                print('nop')
        action = ItemAction(self, self._getMnemonic(description), item, self.service.item_selected)
        self.addAction(action)

    def _add_items_to_self(self, items, on_desktop):
        if False:
            return 10
        if cm.ConfigManager.SETTINGS[cm_constants.SORT_BY_USAGE_COUNT]:
            items.sort(key=lambda obj: obj.usageCount, reverse=True)
        else:
            items.sort(key=lambda obj: str(obj))
        for item in items:
            if on_desktop:
                self._add_item(item.get_description(self.service.lastStackState), item)
            else:
                self._add_item(item.description, item)

    def _getMnemonic(self, desc):
        if False:
            print('Hello World!')
        return desc

class SubMenu(QAction):
    """
    This QAction is used to create submenu in the popup menu.
    It gets used when a folder with a sub-folder has a
    hotkey assigned, to recursively show subfolder contents.
    """

    def __init__(self, title: str, parent: PopupMenu, service, folders: FolderList=None, items: List[Item]=None, on_desktop: bool=True):
        if False:
            for i in range(10):
                print('nop')
        icon = QIcon.fromTheme('folder')
        super(SubMenu, self).__init__(icon, title, parent)
        self.setMenu(PopupMenu(service, folders, items, on_desktop, title, parent))

    def setParent(self, parent: QWidget=None):
        if False:
            print('Hello World!')
        super(SubMenu, self).setParent(parent)
        self.menu().setParent(parent)

class ItemAction(QAction):
    action_sig = pyqtSignal([autokey.model.abstract_hotkey.AbstractHotkey], name='action_sig')

    def __init__(self, parent: QWidget, description: str, item: Item, target):
        if False:
            i = 10
            return i + 15
        icon = ItemAction._icon_for_item(item)
        super(ItemAction, self).__init__(icon, description, parent)
        self.item = item
        self.triggered.connect(lambda : self.action_sig.emit(self.item))
        self.action_sig.connect(target)

    @staticmethod
    def _icon_for_item(item: Item) -> QIcon:
        if False:
            while True:
                i = 10
        if isinstance(item, autokey.model.script.Script):
            return QIcon.fromTheme('text-x-python')
        elif isinstance(item, autokey.model.phrase.Phrase):
            return QIcon.fromTheme('text-x-generic')
        else:
            error_msg = "ItemAction got unknown item. Expected Union[autokey.model.script.Script, autokey.model.phrase.Phrase], got '{}'".format(str(type(item)))
            logger.error(error_msg)
            raise ValueError(error_msg)