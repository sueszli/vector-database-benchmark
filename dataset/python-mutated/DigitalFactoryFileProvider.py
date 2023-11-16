import os
from UM.FileProvider import FileProvider
from UM.Logger import Logger
from cura.API import Account
from cura.CuraApplication import CuraApplication
from .DigitalFactoryController import DigitalFactoryController

class DigitalFactoryFileProvider(FileProvider):

    def __init__(self, df_controller: DigitalFactoryController) -> None:
        if False:
            return 10
        super().__init__()
        self._controller = df_controller
        self.menu_item_display_text = 'From Digital Library'
        self.shortcut = 'Ctrl+Shift+O'
        plugin_path = os.path.dirname(os.path.dirname(__file__))
        self._dialog_path = os.path.join(plugin_path, 'resources', 'qml', 'DigitalFactoryOpenDialog.qml')
        self._dialog = None
        self._account = CuraApplication.getInstance().getCuraAPI().account
        self._controller.userAccessStateChanged.connect(self._onUserAccessStateChanged)
        self.enabled = self._account.isLoggedIn and self._controller.userAccountHasLibraryAccess()
        self.priority = 10

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Function called every time the 'From Digital Factory' option of the 'Open File(s)' submenu is triggered\n        "
        self.loadWindow()
        if self._account.isLoggedIn and self._controller.userAccountHasLibraryAccess():
            self._controller.initialize()
            if not self._dialog:
                Logger.log('e', 'Unable to create the Digital Library Open dialog.')
                return
            self._dialog.show()

    def loadWindow(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create the GUI window for the Digital Library Open dialog. If the window is already open, bring the focus on it.\n        '
        if self._dialog:
            self._dialog.requestActivate()
            return
        self._dialog = CuraApplication.getInstance().createQmlComponent(self._dialog_path, {'manager': self._controller})
        if not self._dialog:
            Logger.log('e', 'Unable to create the Digital Library Open dialog.')

    def _onUserAccessStateChanged(self, logged_in: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Sets the enabled status of the DigitalFactoryFileProvider according to the account's login status\n        :param logged_in: The new login status\n        "
        self.enabled = logged_in and self._controller.userAccountHasLibraryAccess()
        self.enabledChanged.emit()