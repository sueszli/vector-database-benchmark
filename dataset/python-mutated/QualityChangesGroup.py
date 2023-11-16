from typing import Any, Dict, Optional
from PyQt6.QtCore import QObject, pyqtProperty, pyqtSignal

class QualityChangesGroup(QObject):
    """Data struct to group several quality changes instance containers together.

    Each group represents one "custom profile" as the user sees it, which contains an instance container for the
    global stack and one instance container per extruder.
    """

    def __init__(self, name: str, quality_type: str, intent_category: str, parent: Optional['QObject']=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._name = name
        self.quality_type = quality_type
        self.intent_category = intent_category
        self.is_available = False
        self.metadata_for_global = {}
        self.metadata_per_extruder = {}
    nameChanged = pyqtSignal()

    def setName(self, name: str) -> None:
        if False:
            while True:
                i = 10
        if self._name != name:
            self._name = name
            self.nameChanged.emit()

    @pyqtProperty(str, fset=setName, notify=nameChanged)
    def name(self) -> str:
        if False:
            return 10
        return self._name

    def __str__(self) -> str:
        if False:
            return 10
        return '{class_name}[{name}, available = {is_available}]'.format(class_name=self.__class__.__name__, name=self.name, is_available=self.is_available)