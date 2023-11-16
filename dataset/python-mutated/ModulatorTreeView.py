from PyQt5.QtWidgets import QTreeView
from PyQt5.QtCore import pyqtSignal, QItemSelectionModel
from urh.models import GeneratorTreeModel

class ModulatorTreeView(QTreeView):
    selection_changed = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)

    def model(self) -> GeneratorTreeModel:
        if False:
            print('Hello World!')
        return super().model()

    def selectionModel(self) -> QItemSelectionModel:
        if False:
            i = 10
            return i + 15
        return super().selectionModel()

    def selectionChanged(self, QItemSelection, QItemSelection_1):
        if False:
            for i in range(10):
                print('nop')
        self.selection_changed.emit()
        super().selectionChanged(QItemSelection, QItemSelection_1)