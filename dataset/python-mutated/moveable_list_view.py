from functools import partial
from PyQt6 import QtCore, QtWidgets

class MoveableListView:

    def __init__(self, list_widget, up_button, down_button, callback=None):
        if False:
            return 10
        self.list_widget = list_widget
        self.up_button = up_button
        self.down_button = down_button
        self.update_callback = callback
        self.up_button.clicked.connect(partial(self.move_item, 1))
        self.down_button.clicked.connect(partial(self.move_item, -1))
        self.list_widget.currentRowChanged.connect(self.update_buttons)
        self.list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DragDrop)
        self.list_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)

    def move_item(self, offset):
        if False:
            i = 10
            return i + 15
        current_index = self.list_widget.currentRow()
        offset_index = current_index - offset
        offset_item = self.list_widget.item(offset_index)
        if offset_item:
            current_item = self.list_widget.takeItem(current_index)
            self.list_widget.insertItem(offset_index, current_item)
            self.list_widget.setCurrentItem(current_item)
            self.update_buttons()

    def update_buttons(self):
        if False:
            print('Hello World!')
        current_row = self.list_widget.currentRow()
        self.up_button.setEnabled(current_row > 0)
        self.down_button.setEnabled(current_row < self.list_widget.count() - 1)
        if self.update_callback:
            self.update_callback()