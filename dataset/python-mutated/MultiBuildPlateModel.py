from PyQt6.QtCore import QTimer, pyqtSignal, pyqtProperty
from UM.Application import Application
from UM.Scene.Camera import Camera
from UM.Scene.Selection import Selection
from UM.Qt.ListModel import ListModel

class MultiBuildPlateModel(ListModel):
    """This is the model for multi build plate feature.

    This has nothing to do with the build plate types you can choose on the sidebar for a machine.
    """
    maxBuildPlateChanged = pyqtSignal()
    activeBuildPlateChanged = pyqtSignal()
    selectionChanged = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._update_timer = QTimer()
        self._update_timer.setInterval(100)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._updateSelectedObjectBuildPlateNumbers)
        self._application = Application.getInstance()
        self._application.getController().getScene().sceneChanged.connect(self._updateSelectedObjectBuildPlateNumbersDelayed)
        Selection.selectionChanged.connect(self._updateSelectedObjectBuildPlateNumbers)
        self._max_build_plate = 1
        self._active_build_plate = -1

    def setMaxBuildPlate(self, max_build_plate):
        if False:
            while True:
                i = 10
        if self._max_build_plate != max_build_plate:
            self._max_build_plate = max_build_plate
            self.maxBuildPlateChanged.emit()

    @pyqtProperty(int, notify=maxBuildPlateChanged)
    def maxBuildPlate(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the highest build plate number'
        return self._max_build_plate

    def setActiveBuildPlate(self, nr):
        if False:
            for i in range(10):
                print('nop')
        if self._active_build_plate != nr:
            self._active_build_plate = nr
            self.activeBuildPlateChanged.emit()

    @pyqtProperty(int, notify=activeBuildPlateChanged)
    def activeBuildPlate(self):
        if False:
            while True:
                i = 10
        return self._active_build_plate

    def _updateSelectedObjectBuildPlateNumbersDelayed(self, *args):
        if False:
            return 10
        if not isinstance(args[0], Camera):
            self._update_timer.start()

    def _updateSelectedObjectBuildPlateNumbers(self, *args):
        if False:
            i = 10
            return i + 15
        result = set()
        for node in Selection.getAllSelectedObjects():
            result.add(node.callDecoration('getBuildPlateNumber'))
        self._selection_build_plates = list(result)
        self.selectionChanged.emit()

    @pyqtProperty('QVariantList', notify=selectionChanged)
    def selectionBuildPlates(self):
        if False:
            for i in range(10):
                print('nop')
        return self._selection_build_plates