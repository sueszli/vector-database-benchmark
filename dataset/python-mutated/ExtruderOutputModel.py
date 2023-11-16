from typing import Optional, TYPE_CHECKING
from PyQt6.QtCore import pyqtSignal, pyqtProperty, QObject, pyqtSlot
from .ExtruderConfigurationModel import ExtruderConfigurationModel
if TYPE_CHECKING:
    from .MaterialOutputModel import MaterialOutputModel
    from .PrinterOutputModel import PrinterOutputModel

class ExtruderOutputModel(QObject):
    targetHotendTemperatureChanged = pyqtSignal()
    hotendTemperatureChanged = pyqtSignal()
    extruderConfigurationChanged = pyqtSignal()
    isPreheatingChanged = pyqtSignal()

    def __init__(self, printer: 'PrinterOutputModel', position: int, parent=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._printer = printer
        self._position = position
        self._target_hotend_temperature = 0.0
        self._hotend_temperature = 0.0
        self._is_preheating = False
        self._extruder_configuration = ExtruderConfigurationModel()
        self._extruder_configuration.position = self._position
        self._extruder_configuration.extruderConfigurationChanged.connect(self.extruderConfigurationChanged)

    def getPrinter(self) -> 'PrinterOutputModel':
        if False:
            return 10
        return self._printer

    def getPosition(self) -> int:
        if False:
            print('Hello World!')
        return self._position

    @pyqtProperty(bool, constant=True)
    def canPreHeatHotends(self) -> bool:
        if False:
            return 10
        if self._printer:
            return self._printer.canPreHeatHotends
        return False

    @pyqtProperty(QObject, notify=extruderConfigurationChanged)
    def activeMaterial(self) -> Optional['MaterialOutputModel']:
        if False:
            while True:
                i = 10
        return self._extruder_configuration.activeMaterial

    def updateActiveMaterial(self, material: Optional['MaterialOutputModel']) -> None:
        if False:
            return 10
        self._extruder_configuration.setMaterial(material)

    def updateHotendTemperature(self, temperature: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the hotend temperature. This only changes it locally.'
        if self._hotend_temperature != temperature:
            self._hotend_temperature = temperature
            self.hotendTemperatureChanged.emit()

    def updateTargetHotendTemperature(self, temperature: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._target_hotend_temperature != temperature:
            self._target_hotend_temperature = temperature
            self.targetHotendTemperatureChanged.emit()

    @pyqtSlot(float)
    def setTargetHotendTemperature(self, temperature: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set the target hotend temperature. This ensures that it's actually sent to the remote."
        self._printer.getController().setTargetHotendTemperature(self._printer, self, temperature)
        self.updateTargetHotendTemperature(temperature)

    @pyqtProperty(float, notify=targetHotendTemperatureChanged)
    def targetHotendTemperature(self) -> float:
        if False:
            print('Hello World!')
        return self._target_hotend_temperature

    @pyqtProperty(float, notify=hotendTemperatureChanged)
    def hotendTemperature(self) -> float:
        if False:
            print('Hello World!')
        return self._hotend_temperature

    @pyqtProperty(str, notify=extruderConfigurationChanged)
    def hotendID(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._extruder_configuration.hotendID

    def updateHotendID(self, hotend_id: str) -> None:
        if False:
            while True:
                i = 10
        self._extruder_configuration.setHotendID(hotend_id)

    @pyqtProperty(QObject, notify=extruderConfigurationChanged)
    def extruderConfiguration(self) -> Optional[ExtruderConfigurationModel]:
        if False:
            while True:
                i = 10
        if self._extruder_configuration.isValid():
            return self._extruder_configuration
        return None

    def updateIsPreheating(self, pre_heating: bool) -> None:
        if False:
            print('Hello World!')
        if self._is_preheating != pre_heating:
            self._is_preheating = pre_heating
            self.isPreheatingChanged.emit()

    @pyqtProperty(bool, notify=isPreheatingChanged)
    def isPreheating(self) -> bool:
        if False:
            while True:
                i = 10
        return self._is_preheating

    @pyqtSlot(float, float)
    def preheatHotend(self, temperature: float, duration: float) -> None:
        if False:
            i = 10
            return i + 15
        'Pre-heats the extruder before printer.\n\n        :param temperature: The temperature to heat the extruder to, in degrees\n            Celsius.\n        :param duration: How long the bed should stay warm, in seconds.\n        '
        self._printer._controller.preheatHotend(self, temperature, duration)

    @pyqtSlot()
    def cancelPreheatHotend(self) -> None:
        if False:
            return 10
        self._printer._controller.cancelPreheatHotend(self)