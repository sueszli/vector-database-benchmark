import sys
from qtpy.QtCore import Qt, QPoint
from qtpy.QtWidgets import QListWidget, QListWidgetItem
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.utils.icon_manager import ima

class TabSwitcherWidget(QListWidget, SpyderConfigurationAccessor):
    """Show tabs in mru order and change between them."""
    CONF_SECTION = 'editor'

    def __init__(self, parent, stack_history, tabs):
        if False:
            for i in range(10):
                print('nop')
        QListWidget.__init__(self, parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog)
        self.editor = parent
        self.stack_history = stack_history
        self.tabs = tabs
        self.setSelectionMode(QListWidget.SingleSelection)
        self.itemActivated.connect(self.item_selected)
        self.id_list = []
        self.load_data()
        size = self.get_conf('completion/size', section='main')
        self.resize(*size)
        self.set_dialog_position()
        self.setCurrentRow(0)
        self.config_shortcut(lambda : self.select_row(-1), context='Editor', name='Go to previous file', parent=self)
        self.config_shortcut(lambda : self.select_row(1), context='Editor', name='Go to next file', parent=self)

    def load_data(self):
        if False:
            print('Hello World!')
        'Fill ListWidget with the tabs texts.\n\n        Add elements in inverse order of stack_history.\n        '
        for index in reversed(self.stack_history):
            text = self.tabs.tabText(index)
            text = text.replace('&', '')
            item = QListWidgetItem(ima.icon('TextFileIcon'), text)
            self.addItem(item)

    def item_selected(self, item=None):
        if False:
            i = 10
            return i + 15
        'Change to the selected document and hide this widget.'
        if item is None:
            item = self.currentItem()
        try:
            index = self.stack_history[-(self.currentRow() + 1)]
        except IndexError:
            pass
        else:
            self.editor.set_stack_index(index)
            self.editor.current_changed(index)
        self.hide()

    def select_row(self, steps):
        if False:
            while True:
                i = 10
        'Move selected row a number of steps.\n\n        Iterates in a cyclic behaviour.\n        '
        row = (self.currentRow() + steps) % self.count()
        self.setCurrentRow(row)

    def set_dialog_position(self):
        if False:
            return 10
        'Positions the tab switcher in the top-center of the editor.'
        left = int(self.editor.geometry().width() / 2 - self.width() / 2)
        top = int(self.editor.tabs.tabBar().geometry().height() + self.editor.fname_label.geometry().height())
        self.move(self.editor.mapToGlobal(QPoint(left, top)))

    def keyReleaseEvent(self, event):
        if False:
            i = 10
            return i + 15
        'Reimplement Qt method.\n\n        Handle "most recent used" tab behavior,\n        When ctrl is released and tab_switcher is visible, tab will be changed.\n        '
        if self.isVisible():
            qsc = self.get_shortcut(context='Editor', name='Go to next file')
            for key in qsc.split('+'):
                key = key.lower()
                if key == 'ctrl' and event.key() == Qt.Key_Control or (key == 'alt' and event.key() == Qt.Key_Alt):
                    self.item_selected()
        event.accept()

    def keyPressEvent(self, event):
        if False:
            while True:
                i = 10
        'Reimplement Qt method to allow cyclic behavior.'
        if event.key() == Qt.Key_Down:
            self.select_row(1)
        elif event.key() == Qt.Key_Up:
            self.select_row(-1)

    def focusOutEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Reimplement Qt method to close the widget when loosing focus.'
        event.ignore()
        if sys.platform == 'darwin':
            if event.reason() != Qt.ActiveWindowFocusReason:
                self.close()
        else:
            self.close()