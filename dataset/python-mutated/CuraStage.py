from PyQt6.QtCore import pyqtProperty, QUrl
from UM.Stage import Stage

class CuraStage(Stage):

    def __init__(self, parent=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)

    @pyqtProperty(str, constant=True)
    def stageId(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.getPluginId()

    @pyqtProperty(QUrl, constant=True)
    def mainComponent(self) -> QUrl:
        if False:
            for i in range(10):
                print('nop')
        return self.getDisplayComponent('main')

    @pyqtProperty(QUrl, constant=True)
    def stageMenuComponent(self) -> QUrl:
        if False:
            print('Hello World!')
        return self.getDisplayComponent('menu')
__all__ = ['CuraStage']