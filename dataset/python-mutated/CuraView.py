from PyQt6.QtCore import pyqtProperty, QUrl
from UM.Resources import Resources
from UM.View.View import View
from cura.CuraApplication import CuraApplication

class CuraView(View):

    def __init__(self, parent=None, use_empty_menu_placeholder: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._empty_menu_placeholder_url = QUrl.fromLocalFile(Resources.getPath(CuraApplication.ResourceTypes.QmlFiles, 'EmptyViewMenuComponent.qml'))
        self._use_empty_menu_placeholder = use_empty_menu_placeholder

    @pyqtProperty(QUrl, constant=True)
    def mainComponent(self) -> QUrl:
        if False:
            for i in range(10):
                print('nop')
        return self.getDisplayComponent('main')

    @pyqtProperty(QUrl, constant=True)
    def stageMenuComponent(self) -> QUrl:
        if False:
            i = 10
            return i + 15
        url = self.getDisplayComponent('menu')
        if not url.toString() and self._use_empty_menu_placeholder:
            url = self._empty_menu_placeholder_url
        return url