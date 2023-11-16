from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QRadioButton
from qt.util import horizontal_spacer

class RadioBox(QWidget):

    def __init__(self, parent=None, items=None, spread=True, **kwargs):
        if False:
            i = 10
            return i + 15
        if items is None:
            items = []
        super().__init__(parent, **kwargs)
        self._buttons = []
        self._labels = items
        self._selected_index = 0
        self._spacer = horizontal_spacer() if not spread else None
        self._layout = QHBoxLayout(self)
        self._update_buttons()

    def _update_buttons(self):
        if False:
            print('Hello World!')
        if self._spacer is not None:
            self._layout.removeItem(self._spacer)
        to_remove = self._buttons[len(self._labels):]
        for button in to_remove:
            self._layout.removeWidget(button)
            button.setParent(None)
        del self._buttons[len(self._labels):]
        to_add = self._labels[len(self._buttons):]
        for _ in to_add:
            button = QRadioButton(self)
            self._buttons.append(button)
            self._layout.addWidget(button)
            button.toggled.connect(self.buttonToggled)
        if self._spacer is not None:
            self._layout.addItem(self._spacer)
        if not self._buttons:
            return
        for (button, label) in zip(self._buttons, self._labels):
            button.setText(label)
        self._update_selection()

    def _update_selection(self):
        if False:
            while True:
                i = 10
        self._selected_index = max(0, min(self._selected_index, len(self._buttons) - 1))
        selected = self._buttons[self._selected_index]
        selected.setChecked(True)

    def buttonToggled(self):
        if False:
            print('Hello World!')
        for (i, button) in enumerate(self._buttons):
            if button.isChecked():
                self._selected_index = i
                self.itemSelected.emit(i)
                break
    itemSelected = pyqtSignal(int)

    @property
    def buttons(self):
        if False:
            i = 10
            return i + 15
        return self._buttons[:]

    @property
    def items(self):
        if False:
            i = 10
            return i + 15
        return self._labels[:]

    @items.setter
    def items(self, value):
        if False:
            print('Hello World!')
        self._labels = value
        self._update_buttons()

    @property
    def selected_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self._selected_index

    @selected_index.setter
    def selected_index(self, value):
        if False:
            return 10
        self._selected_index = value
        self._update_selection()