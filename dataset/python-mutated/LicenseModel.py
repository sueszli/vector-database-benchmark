from PyQt6.QtCore import QObject, pyqtProperty, pyqtSignal
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class LicenseModel(QObject):
    DEFAULT_DECLINE_BUTTON_TEXT = catalog.i18nc('@button', 'Decline')
    ACCEPT_BUTTON_TEXT = catalog.i18nc('@button', 'Agree')
    dialogTitleChanged = pyqtSignal()
    packageNameChanged = pyqtSignal()
    licenseTextChanged = pyqtSignal()
    iconChanged = pyqtSignal()

    def __init__(self, decline_button_text: str=DEFAULT_DECLINE_BUTTON_TEXT, parent=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._current_page_idx = 0
        self._page_count = 1
        self._dialogTitle = ''
        self._license_text = ''
        self._package_name = ''
        self._icon_url = ''
        self._decline_button_text = decline_button_text

    @pyqtProperty(str, constant=True)
    def acceptButtonText(self):
        if False:
            while True:
                i = 10
        return self.ACCEPT_BUTTON_TEXT

    @pyqtProperty(str, constant=True)
    def declineButtonText(self):
        if False:
            return 10
        return self._decline_button_text

    @pyqtProperty(str, notify=dialogTitleChanged)
    def dialogTitle(self) -> str:
        if False:
            while True:
                i = 10
        return self._dialogTitle

    @pyqtProperty(str, notify=packageNameChanged)
    def packageName(self) -> str:
        if False:
            return 10
        return self._package_name

    def setPackageName(self, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._package_name = name
        self.packageNameChanged.emit()

    @pyqtProperty(str, notify=iconChanged)
    def iconUrl(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._icon_url

    def setIconUrl(self, url: str):
        if False:
            return 10
        self._icon_url = url
        self.iconChanged.emit()

    @pyqtProperty(str, notify=licenseTextChanged)
    def licenseText(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._license_text

    def setLicenseText(self, license_text: str) -> None:
        if False:
            return 10
        if self._license_text != license_text:
            self._license_text = license_text
            self.licenseTextChanged.emit()

    def setCurrentPageIdx(self, idx: int) -> None:
        if False:
            print('Hello World!')
        self._current_page_idx = idx
        self._updateDialogTitle()

    def setPageCount(self, count: int) -> None:
        if False:
            while True:
                i = 10
        self._page_count = count
        self._updateDialogTitle()

    def _updateDialogTitle(self):
        if False:
            return 10
        self._dialogTitle = catalog.i18nc('@title:window', 'Plugin License Agreement')
        if self._page_count > 1:
            self._dialogTitle = self._dialogTitle + ' ({}/{})'.format(self._current_page_idx + 1, self._page_count)
        self.dialogTitleChanged.emit()