from typing import Union, List, Optional
from PyQt5.QtCore import Qt, QEvent, QModelIndex
from PyQt5.QtGui import QKeySequence, QIcon, QKeyEvent, QMouseEvent, QDragMoveEvent, QDropEvent
from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QAbstractItemView
import autokey.model.folder
import autokey.model.phrase
import autokey.model.script

class AkTreeWidget(QTreeWidget):

    def edit(self, index: QModelIndex, trigger: QAbstractItemView.EditTrigger, event: QEvent):
        if False:
            for i in range(10):
                print('nop')
        if index.column() == 0:
            super(QTreeWidget, self).edit(index, trigger, event)
        return False

    def keyPressEvent(self, event: QKeyEvent):
        if False:
            return 10
        if self.window().is_dirty() and (event.matches(QKeySequence.MoveToNextLine) or event.matches(QKeySequence.MoveToPreviousLine)):
            veto = self.window().central_widget.promptToSave()
            if not veto:
                QTreeWidget.keyPressEvent(self, event)
            else:
                event.ignore()
        else:
            QTreeWidget.keyPressEvent(self, event)

    def mousePressEvent(self, event: QMouseEvent):
        if False:
            while True:
                i = 10
        if self.window().is_dirty():
            veto = self.window().central_widget.promptToSave()
            if not veto:
                QTreeWidget.mousePressEvent(self, event)
                QTreeWidget.mouseReleaseEvent(self, event)
            else:
                event.ignore()
        else:
            QTreeWidget.mousePressEvent(self, event)

    def dragMoveEvent(self, event: QDragMoveEvent):
        if False:
            print('Hello World!')
        target = self.itemAt(event.pos())
        if isinstance(target, FolderWidgetItem):
            QTreeWidget.dragMoveEvent(self, event)
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if False:
            return 10
        target = self.itemAt(event.pos())
        sources = self.selectedItems()
        self.window().central_widget.move_items(sources, target)

class FolderWidgetItem(QTreeWidgetItem):

    def __init__(self, parent: Optional[QTreeWidgetItem], folder: autokey.model.folder.Folder):
        if False:
            return 10
        QTreeWidgetItem.__init__(self)
        self.folder = folder
        self.setIcon(0, QIcon.fromTheme('folder'))
        self.setText(0, folder.title)
        self.setText(1, folder.get_abbreviations())
        self.setText(2, folder.get_hotkey_string())
        self.setData(3, Qt.UserRole, folder)
        if parent is not None:
            parent.addChild(self)
        self.setFlags(self.flags() | Qt.ItemIsEditable)

    def update(self):
        if False:
            print('Hello World!')
        self.setText(0, self.folder.title)
        self.setText(1, self.folder.get_abbreviations())
        self.setText(2, self.folder.get_hotkey_string())

    def __ge__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, ScriptWidgetItem):
            return QTreeWidgetItem.__ge__(self, other)
        else:
            return False

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, FolderWidgetItem):
            return QTreeWidgetItem.__lt__(self, other)
        else:
            return True

class PhraseWidgetItem(QTreeWidgetItem):

    def __init__(self, parent: Optional[FolderWidgetItem], phrase: autokey.model.phrase.Phrase):
        if False:
            for i in range(10):
                print('nop')
        QTreeWidgetItem.__init__(self)
        self.phrase = phrase
        self.setIcon(0, QIcon.fromTheme('text-x-generic'))
        self.setText(0, phrase.description)
        self.setText(1, phrase.get_abbreviations())
        self.setText(2, phrase.get_hotkey_string())
        self.setData(3, Qt.UserRole, phrase)
        if parent is not None:
            parent.addChild(self)
        self.setFlags(self.flags() | Qt.ItemIsEditable)

    def update(self):
        if False:
            i = 10
            return i + 15
        self.setText(0, self.phrase.description)
        self.setText(1, self.phrase.get_abbreviations())
        self.setText(2, self.phrase.get_hotkey_string())

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        if isinstance(other, ScriptWidgetItem):
            return QTreeWidgetItem.__ge__(self, other)
        else:
            return True

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, PhraseWidgetItem):
            return QTreeWidgetItem.__lt__(self, other)
        else:
            return False

class ScriptWidgetItem(QTreeWidgetItem):

    def __init__(self, parent: Optional[FolderWidgetItem], script: autokey.model.script.Script):
        if False:
            return 10
        QTreeWidgetItem.__init__(self)
        self.script = script
        self.setIcon(0, QIcon.fromTheme('text-x-python'))
        self.setText(0, script.description)
        self.setText(1, script.get_abbreviations())
        self.setText(2, script.get_hotkey_string())
        self.setData(3, Qt.UserRole, script)
        if parent is not None:
            parent.addChild(self)
        self.setFlags(self.flags() | Qt.ItemIsEditable)

    def update(self):
        if False:
            i = 10
            return i + 15
        self.setText(0, self.script.description)
        self.setText(1, self.script.get_abbreviations())
        self.setText(2, self.script.get_hotkey_string())

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        if isinstance(other, ScriptWidgetItem):
            return QTreeWidgetItem.__ge__(self, other)
        else:
            return True

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, ScriptWidgetItem):
            return QTreeWidgetItem.__lt__(self, other)
        else:
            return False
ItemType = Union[autokey.model.folder.Folder, autokey.model.phrase.Phrase, autokey.model.script.Script]
ItemWidgetType = Union[FolderWidgetItem, PhraseWidgetItem, ScriptWidgetItem]

class WidgetItemFactory:

    def __init__(self, root_folders: List[autokey.model.folder.Folder]):
        if False:
            i = 10
            return i + 15
        self.folders = root_folders

    def get_root_folder_list(self):
        if False:
            return 10
        root_items = []
        for folder in self.folders:
            item = WidgetItemFactory._build_item(None, folder)
            root_items.append(item)
            WidgetItemFactory.process_folder(item, folder)
        return root_items

    @staticmethod
    def process_folder(parent_item: ItemWidgetType, parent_folder: autokey.model.folder.Folder):
        if False:
            while True:
                i = 10
        for folder in parent_folder.folders:
            item = WidgetItemFactory._build_item(parent_item, folder)
            WidgetItemFactory.process_folder(item, folder)
        for childModelItem in parent_folder.items:
            WidgetItemFactory._build_item(parent_item, childModelItem)

    @staticmethod
    def _build_item(parent: Optional[FolderWidgetItem], item: ItemType) -> ItemWidgetType:
        if False:
            return 10
        if isinstance(item, autokey.model.folder.Folder):
            return FolderWidgetItem(parent, item)
        elif isinstance(item, autokey.model.phrase.Phrase):
            return PhraseWidgetItem(parent, item)
        elif isinstance(item, autokey.model.script.Script):
            return ScriptWidgetItem(parent, item)