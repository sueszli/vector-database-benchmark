from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal, pyqtProperty
from UM.TaskManagement.HttpRequestManager import HttpRequestManager

class ConnectionStatus(QObject):
    """Provides an estimation of whether internet is reachable

    Estimation is updated with every request through HttpRequestManager.
    Acts as a proxy to HttpRequestManager.internetReachableChanged without
    exposing the HttpRequestManager in its entirety.
    """
    __instance = None
    internetReachableChanged = pyqtSignal()

    @classmethod
    def getInstance(cls, *args, **kwargs) -> 'ConnectionStatus':
        if False:
            while True:
                i = 10
        if cls.__instance is None:
            cls.__instance = cls(*args, **kwargs)
        return cls.__instance

    def __init__(self, parent: Optional['QObject']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        manager = HttpRequestManager.getInstance()
        self._is_internet_reachable = manager.isInternetReachable
        manager.internetReachableChanged.connect(self._onInternetReachableChanged)

    @pyqtProperty(bool, notify=internetReachableChanged)
    def isInternetReachable(self) -> bool:
        if False:
            while True:
                i = 10
        return self._is_internet_reachable

    def _onInternetReachableChanged(self, reachable: bool):
        if False:
            return 10
        if reachable != self._is_internet_reachable:
            self._is_internet_reachable = reachable
            self.internetReachableChanged.emit()