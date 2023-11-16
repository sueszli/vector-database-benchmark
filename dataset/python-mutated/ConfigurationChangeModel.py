from PyQt6.QtCore import pyqtProperty, QObject
BLOCKING_CHANGE_TYPES = ['material_insert', 'buildplate_change']

class ConfigurationChangeModel(QObject):

    def __init__(self, type_of_change: str, index: int, target_name: str, origin_name: str) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._type_of_change = type_of_change
        self._can_override = self._type_of_change not in BLOCKING_CHANGE_TYPES
        self._index = index
        self._target_name = target_name
        self._origin_name = origin_name

    @pyqtProperty(int, constant=True)
    def index(self) -> int:
        if False:
            print('Hello World!')
        return self._index

    @pyqtProperty(str, constant=True)
    def typeOfChange(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._type_of_change

    @pyqtProperty(str, constant=True)
    def targetName(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._target_name

    @pyqtProperty(str, constant=True)
    def originName(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._origin_name

    @pyqtProperty(bool, constant=True)
    def canOverride(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._can_override