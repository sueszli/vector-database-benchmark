import socket
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSlot

class NetworkingUtil(QObject):

    def __init__(self, parent: Optional['QObject']=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)

    @pyqtSlot(str, result=bool)
    def isIPv4(self, address: str) -> bool:
        if False:
            i = 10
            return i + 15
        try:
            socket.inet_pton(socket.AF_INET, address)
            result = True
        except:
            result = False
        return result

    @pyqtSlot(str, result=bool)
    def isIPv6(self, address: str) -> bool:
        if False:
            while True:
                i = 10
        try:
            socket.inet_pton(socket.AF_INET6, address)
            result = True
        except:
            result = False
        return result

    @pyqtSlot(str, result=bool)
    def isValidIP(self, address: str) -> bool:
        if False:
            print('Hello World!')
        return self.isIPv4(address) or self.isIPv6(address)
__all__ = ['NetworkingUtil']