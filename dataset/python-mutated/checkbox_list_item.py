from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QListWidgetItem

class CheckboxListItem(QListWidgetItem):

    def __init__(self, text='', checked=False, data=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.setText(text)
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        self.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self.data = data

    @property
    def checked(self):
        if False:
            while True:
                i = 10
        return self.checkState() == Qt.CheckState.Checked