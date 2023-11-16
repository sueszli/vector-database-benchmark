from PyQt5.QtGui import QMouseEvent
from PyQt5.QtCore import Qt, pyqtSlot, QModelIndex, pyqtSignal
from urh.models.SimulatorMessageFieldModel import SimulatorMessageFieldModel
from urh.ui.views.ProtocolLabelTableView import ProtocolLabelTableView

class SimulatorLabelTableView(ProtocolLabelTableView):
    item_link_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.setMouseTracking(True)
        self.clicked.connect(self.on_clicked)

    def model(self) -> SimulatorMessageFieldModel:
        if False:
            return 10
        return super().model()

    def mouseMoveEvent(self, e: QMouseEvent):
        if False:
            return 10
        index = self.indexAt(e.pos())
        if self.model().link_index(index):
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.unsetCursor()

    @pyqtSlot(QModelIndex)
    def on_clicked(self, index: QModelIndex):
        if False:
            for i in range(10):
                print('nop')
        if self.model().link_index(index):
            self.item_link_clicked.emit(index.row(), index.column())