from PyQt5.QtWidgets import QTreeView, QAbstractItemView
from PyQt5.QtCore import QItemSelectionModel
from urh.models.GeneratorTreeModel import GeneratorTreeModel

class GeneratorTreeView(QTreeView):

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def model(self) -> GeneratorTreeModel:
        if False:
            while True:
                i = 10
        return super().model()

    def selectionModel(self) -> QItemSelectionModel:
        if False:
            i = 10
            return i + 15
        return super().selectionModel()