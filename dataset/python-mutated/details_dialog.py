from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QAbstractItemView, QSizePolicy, QGridLayout, QSplitter, QFrame
from PyQt5.QtGui import QResizeEvent
from hscommon.trans import trget
from qt.details_dialog import DetailsDialog as DetailsDialogBase
from qt.details_table import DetailsTable
from qt.pe.image_viewer import ViewerToolBar, ScrollAreaImageViewer, ScrollAreaController
tr = trget('ui')

class DetailsDialog(DetailsDialogBase):

    def __init__(self, parent, app):
        if False:
            print('Hello World!')
        self.vController = None
        super().__init__(parent, app)

    def _setupUi(self):
        if False:
            return 10
        self.setWindowTitle(tr('Details'))
        self.resize(502, 502)
        self.setMinimumSize(QSize(250, 250))
        self.splitter = QSplitter(Qt.Vertical)
        self.topFrame = EmittingFrame()
        self.topFrame.setFrameShape(QFrame.StyledPanel)
        self.horizontalLayout = QGridLayout()
        self.horizontalLayout.setColumnMinimumWidth(1, 10)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setColumnStretch(0, 32)
        self.horizontalLayout.setColumnStretch(1, 2)
        self.horizontalLayout.setColumnStretch(2, 32)
        self.horizontalLayout.setRowStretch(0, 1)
        self.horizontalLayout.setRowStretch(1, 24)
        self.horizontalLayout.setRowStretch(2, 1)
        self.horizontalLayout.setSpacing(1)
        self.selectedImageViewer = ScrollAreaImageViewer(self, 'selectedImage')
        self.horizontalLayout.addWidget(self.selectedImageViewer, 0, 0, 3, 1)
        self.vController = ScrollAreaController(self)
        self.verticalToolBar = ViewerToolBar(self, self.vController)
        self.verticalToolBar.setOrientation(Qt.Orientation(Qt.Vertical))
        self.horizontalLayout.addWidget(self.verticalToolBar, 1, 1, 1, 1, Qt.AlignCenter)
        self.referenceImageViewer = ScrollAreaImageViewer(self, 'referenceImage')
        self.horizontalLayout.addWidget(self.referenceImageViewer, 0, 2, 3, 1)
        self.topFrame.setLayout(self.horizontalLayout)
        self.splitter.addWidget(self.topFrame)
        self.splitter.setStretchFactor(0, 8)
        self.tableView = DetailsTable(self)
        size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        self.tableView.setSizePolicy(size_policy)
        self.tableView.setAlternatingRowColors(True)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableView.setShowGrid(False)
        self.splitter.addWidget(self.tableView)
        self.splitter.setStretchFactor(1, 1)
        self.vController.setupViewers(self.selectedImageViewer, self.referenceImageViewer)
        self.setWidget(self.splitter)
        self.topFrame.resized.connect(self.resizeEvent)

    def _update(self):
        if False:
            while True:
                i = 10
        if self.vController is None:
            return
        if not self.app.model.selected_dupes:
            self.vController.resetViewersState()
            return
        dupe = self.app.model.selected_dupes[0]
        group = self.app.model.results.get_group_of_duplicate(dupe)
        ref = group.ref
        self.vController.updateView(ref, dupe, group)

    @pyqtSlot(QResizeEvent)
    def resizeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.ensure_same_sizes()
        if self.vController is None or not self.vController.bestFit:
            return
        self.vController.updateBothImages()

    def show(self):
        if False:
            return 10
        self.tableView.setMaximumHeight(self.tableView.rowHeight(1) * self.tableModel.model.row_count() + self.tableView.verticalHeader().sectionSize(0) + self.splitter.handle(1).size().height())
        DetailsDialogBase.show(self)
        self.ensure_same_sizes()
        self._update()

    def ensure_same_sizes(self):
        if False:
            print('Hello World!')
        if self.selectedImageViewer.size().width() > self.referenceImageViewer.size().width():
            self.selectedImageViewer.resize(self.referenceImageViewer.size())

    def refresh(self):
        if False:
            i = 10
            return i + 15
        DetailsDialogBase.refresh(self)
        if self.isVisible():
            self._update()

class EmittingFrame(QFrame):
    """Emits a signal whenever is resized"""
    resized = pyqtSignal(QResizeEvent)

    def resizeEvent(self, event):
        if False:
            while True:
                i = 10
        self.resized.emit(event)