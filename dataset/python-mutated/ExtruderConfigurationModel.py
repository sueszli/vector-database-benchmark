from typing import Optional
from PyQt6.QtCore import pyqtProperty, QObject, pyqtSignal
from .MaterialOutputModel import MaterialOutputModel

class ExtruderConfigurationModel(QObject):
    extruderConfigurationChanged = pyqtSignal()

    def __init__(self, position: int=-1) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self._position: int = position
        self._material: Optional[MaterialOutputModel] = None
        self._hotend_id: Optional[str] = None

    def setPosition(self, position: int) -> None:
        if False:
            while True:
                i = 10
        self._position = position

    @pyqtProperty(int, fset=setPosition, notify=extruderConfigurationChanged)
    def position(self) -> int:
        if False:
            return 10
        return self._position

    def setMaterial(self, material: Optional[MaterialOutputModel]) -> None:
        if False:
            print('Hello World!')
        if material is None or self._material == material:
            return
        self._material = material
        self.extruderConfigurationChanged.emit()

    @pyqtProperty(QObject, fset=setMaterial, notify=extruderConfigurationChanged)
    def activeMaterial(self) -> Optional[MaterialOutputModel]:
        if False:
            return 10
        return self._material

    @pyqtProperty(QObject, fset=setMaterial, notify=extruderConfigurationChanged)
    def material(self) -> Optional[MaterialOutputModel]:
        if False:
            while True:
                i = 10
        return self._material

    def setHotendID(self, hotend_id: Optional[str]) -> None:
        if False:
            print('Hello World!')
        if self._hotend_id != hotend_id:
            self._hotend_id = ExtruderConfigurationModel.applyNameMappingHotend(hotend_id)
            self.extruderConfigurationChanged.emit()

    @staticmethod
    def applyNameMappingHotend(hotendId) -> str:
        if False:
            for i in range(10):
                print('nop')
        _EXTRUDER_NAME_MAP = {'mk14_hot': '1XA', 'mk14_hot_s': '2XA', 'mk14_c': '1C', 'mk14': '1A', 'mk14_s': '2A'}
        if hotendId in _EXTRUDER_NAME_MAP:
            return _EXTRUDER_NAME_MAP[hotendId]
        return hotendId

    @pyqtProperty(str, fset=setHotendID, notify=extruderConfigurationChanged)
    def hotendID(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._hotend_id

    def isValid(self) -> bool:
        if False:
            i = 10
            return i + 15
        'This method is intended to indicate whether the configuration is valid or not.\n\n        The method checks if the mandatory fields are or not set\n        At this moment is always valid since we allow to have empty material and variants.\n        '
        return True

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        message_chunks = []
        message_chunks.append('Position: ' + str(self._position))
        message_chunks.append('-')
        message_chunks.append('Material: ' + self.activeMaterial.type if self.activeMaterial else 'empty')
        message_chunks.append('-')
        message_chunks.append('HotendID: ' + self.hotendID if self.hotendID else 'empty')
        return ' '.join(message_chunks)

    def __eq__(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(other, ExtruderConfigurationModel):
            return False
        if self._position != other.position:
            return False
        if self.activeMaterial is not None and other.activeMaterial is not None:
            if self.activeMaterial.guid != other.activeMaterial.guid:
                if self.activeMaterial.guid == '' and other.activeMaterial.guid == '':
                    return True
                else:
                    return False
        if self.hotendID != other.hotendID:
            return False
        return True

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash(self._position) ^ (hash(self._material.guid) if self._material is not None else hash(0)) ^ hash(self._hotend_id)