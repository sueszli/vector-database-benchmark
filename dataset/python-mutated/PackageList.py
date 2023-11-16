import tempfile
import json
import os.path
from PyQt6.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, Qt
from typing import cast, Dict, Optional, Set, TYPE_CHECKING
from UM.i18n import i18nCatalog
from UM.Qt.ListModel import ListModel
from UM.TaskManagement.HttpRequestScope import JsonDecoratorScope
from UM.TaskManagement.HttpRequestManager import HttpRequestData, HttpRequestManager
from UM.Logger import Logger
from UM import PluginRegistry
from cura.CuraApplication import CuraApplication
from cura.CuraPackageManager import CuraPackageManager
from cura.UltimakerCloud.UltimakerCloudScope import UltimakerCloudScope
from .PackageModel import PackageModel
from .Constants import USER_PACKAGES_URL, PACKAGES_URL
if TYPE_CHECKING:
    from PyQt6.QtCore import QObject
    from PyQt6.QtNetwork import QNetworkReply
catalog = i18nCatalog('cura')

class PackageList(ListModel):
    """ A List model for Packages, this class serves as parent class for more detailed implementations.
    such as Packages obtained from Remote or Local source
    """
    PackageRole = Qt.ItemDataRole.UserRole + 1
    DISK_WRITE_BUFFER_SIZE = 256 * 1024

    def __init__(self, parent: Optional['QObject']=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._package_manager: CuraPackageManager = cast(CuraPackageManager, CuraApplication.getInstance().getPackageManager())
        self._plugin_registry: PluginRegistry = CuraApplication.getInstance().getPluginRegistry()
        self._account = CuraApplication.getInstance().getCuraAPI().account
        self._error_message = ''
        self.addRoleName(self.PackageRole, 'package')
        self._is_loading = False
        self._has_more = False
        self._has_footer = True
        self._to_install: Dict[str, str] = {}
        self._ongoing_requests: Dict[str, Optional[HttpRequestData]] = {'download_package': None}
        self._scope = JsonDecoratorScope(UltimakerCloudScope(CuraApplication.getInstance()))
        self._license_dialogs: Dict[str, QObject] = {}

    def __del__(self) -> None:
        if False:
            while True:
                i = 10
        ' When this object is deleted it will loop through all registered API requests and aborts them '
        try:
            self.isLoadingChanged.disconnect()
            self.hasMoreChanged.disconnect()
        except RuntimeError:
            pass
        self.cleanUpAPIRequest()

    def abortRequest(self, request_id: str) -> None:
        if False:
            return 10
        'Aborts a single request'
        if request_id in self._ongoing_requests and self._ongoing_requests[request_id]:
            HttpRequestManager.getInstance().abortRequest(self._ongoing_requests[request_id])
            self._ongoing_requests[request_id] = None

    @pyqtSlot()
    def cleanUpAPIRequest(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for request_id in self._ongoing_requests:
            self.abortRequest(request_id)

    @pyqtSlot()
    def updatePackages(self) -> None:
        if False:
            while True:
                i = 10
        ' A Qt slot which will update the List from a source. Actual implementation should be done in the child class'
        pass

    def reset(self) -> None:
        if False:
            while True:
                i = 10
        ' Resets and clears the list'
        self.clear()
    isLoadingChanged = pyqtSignal()

    def setIsLoading(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._is_loading != value:
            self._is_loading = value
            self.isLoadingChanged.emit()

    @pyqtProperty(bool, fset=setIsLoading, notify=isLoadingChanged)
    def isLoading(self) -> bool:
        if False:
            while True:
                i = 10
        ' Indicating if the the packages are loading\n        :return" ``True`` if the list is being obtained, otherwise ``False``\n        '
        return self._is_loading
    hasMoreChanged = pyqtSignal()

    def setHasMore(self, value: bool) -> None:
        if False:
            print('Hello World!')
        if self._has_more != value:
            self._has_more = value
            self.hasMoreChanged.emit()

    @pyqtProperty(bool, fset=setHasMore, notify=hasMoreChanged)
    def hasMore(self) -> bool:
        if False:
            print('Hello World!')
        ' Indicating if there are more packages available to load.\n        :return: ``True`` if there are more packages to load, or ``False``.\n        '
        return self._has_more
    errorMessageChanged = pyqtSignal()

    def setErrorMessage(self, error_message: str) -> None:
        if False:
            return 10
        if self._error_message != error_message:
            self._error_message = error_message
            self.errorMessageChanged.emit()

    @pyqtProperty(str, notify=errorMessageChanged, fset=setErrorMessage)
    def errorMessage(self) -> str:
        if False:
            print('Hello World!')
        ' If an error occurred getting the list of packages, an error message will be held here.\n\n        If no error occurred (yet), this will be an empty string.\n        :return: An error message, if any, or an empty string if everything went okay.\n        '
        return self._error_message

    @pyqtProperty(bool, constant=True)
    def hasFooter(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ' Indicating if the PackageList should have a Footer visible. For paginated PackageLists\n        :return: ``True`` if a Footer should be displayed in the ListView, e.q.: paginated lists, ``False`` Otherwise'
        return self._has_footer

    def getPackageModel(self, package_id: str) -> Optional[PackageModel]:
        if False:
            while True:
                i = 10
        index = self.find('package', package_id)
        data = self.getItem(index)
        if data:
            return data.get('package')
        return None

    def _openLicenseDialog(self, package_id: str, license_content: str) -> None:
        if False:
            return 10
        plugin_path = self._plugin_registry.getPluginPath('Marketplace')
        if plugin_path is None:
            plugin_path = os.path.dirname(__file__)
        license_dialog_component_path = os.path.join(plugin_path, 'resources', 'qml', 'LicenseDialog.qml')
        dialog = CuraApplication.getInstance().createQmlComponent(license_dialog_component_path, {'licenseContent': license_content, 'packageId': package_id, 'handler': self})
        dialog.show()
        self._license_dialogs[package_id] = dialog

    @pyqtSlot(str)
    def onLicenseAccepted(self, package_id: str) -> None:
        if False:
            return 10
        dialog = self._license_dialogs.pop(package_id)
        if dialog is not None:
            dialog.deleteLater()
        self._install(package_id)

    @pyqtSlot(str)
    def onLicenseDeclined(self, package_id: str) -> None:
        if False:
            return 10
        dialog = self._license_dialogs.pop(package_id)
        if dialog is not None:
            dialog.deleteLater()
        self._package_manager.packageInstallingFailed.emit(package_id)

    def _requestInstall(self, package_id: str, update: bool=False) -> None:
        if False:
            return 10
        package_path = self._to_install[package_id]
        license_content = self._package_manager.getPackageLicense(package_path)
        if not update and license_content is not None and (license_content != ''):
            self._openLicenseDialog(package_id, license_content)
        else:
            self._install(package_id, update)

    def _install(self, package_id: str, update: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        package_path = self._to_install.pop(package_id)
        to_be_installed = self._package_manager.installPackage(package_path) is not None
        if not to_be_installed:
            Logger.warning(f'Could not install {package_id}')
            return
        package = self.getPackageModel(package_id)
        if package:
            self.subscribeUserToPackage(package_id, str(package.sdk_version))
        else:
            Logger.log('w', f'Unable to get data on package {package_id}')

    def download(self, package_id: str, url: str, update: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initiate the download request\n\n        :param package_id: the package identification string\n        :param url: the URL from which the package needs to be obtained\n        :param update: A flag if this is download request is an update process\n        '
        if url == '':
            url = f'{PACKAGES_URL}/{package_id}/download'

        def downloadFinished(reply: 'QNetworkReply') -> None:
            if False:
                print('Hello World!')
            self._downloadFinished(package_id, reply, update)

        def downloadError(reply: 'QNetworkReply', error: 'QNetworkReply.NetworkError') -> None:
            if False:
                for i in range(10):
                    print('nop')
            self._downloadError(package_id, update, reply, error)
        self._ongoing_requests['download_package'] = HttpRequestManager.getInstance().get(url, scope=self._scope, callback=downloadFinished, error_callback=downloadError)

    def _downloadFinished(self, package_id: str, reply: 'QNetworkReply', update: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile(mode='wb+', suffix='.curapackage', delete=False) as temp_file:
            try:
                bytes_read = reply.read(self.DISK_WRITE_BUFFER_SIZE)
                while bytes_read:
                    temp_file.write(bytes_read)
                    bytes_read = reply.read(self.DISK_WRITE_BUFFER_SIZE)
            except IOError as e:
                Logger.error(f'Failed to write downloaded package to temp file {e}')
                temp_file.close()
                self._downloadError(package_id, update)
            except RuntimeError:
                temp_file.close()
                return
        temp_file.close()
        self._to_install[package_id] = temp_file.name
        self._ongoing_requests['download_package'] = None
        self._requestInstall(package_id, update)

    def _downloadError(self, package_id: str, update: bool=False, reply: Optional['QNetworkReply']=None, error: Optional['QNetworkReply.NetworkError']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if reply:
            try:
                reply_string = bytes(reply.readAll()).decode()
            except UnicodeDecodeError:
                reply_string = '<error message is corrupt too>'
            Logger.error(f'Failed to download package: {package_id} due to {reply_string}')
        self._package_manager.packageInstallingFailed.emit(package_id)

    def subscribeUserToPackage(self, package_id: str, sdk_version: str) -> None:
        if False:
            i = 10
            return i + 15
        'Subscribe the user (if logged in) to the package for a given SDK\n\n         :param package_id: the package identification string\n         :param sdk_version: the SDK version\n         '
        if self._account.isLoggedIn:
            HttpRequestManager.getInstance().put(url=USER_PACKAGES_URL, data=json.dumps({'data': {'package_id': package_id, 'sdk_version': sdk_version}}).encode(), scope=self._scope)

    def unsunscribeUserFromPackage(self, package_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Unsubscribe the user (if logged in) from the package\n\n         :param package_id: the package identification string\n         '
        if self._account.isLoggedIn:
            HttpRequestManager.getInstance().delete(url=f'{USER_PACKAGES_URL}/{package_id}', scope=self._scope)

    def _connectManageButtonSignals(self, package: PackageModel) -> None:
        if False:
            i = 10
            return i + 15
        package.installPackageTriggered.connect(self.installPackage)
        package.uninstallPackageTriggered.connect(self.uninstallPackage)
        package.updatePackageTriggered.connect(self.updatePackage)

    def installPackage(self, package_id: str, url: str) -> None:
        if False:
            print('Hello World!')
        'Install a package from the Marketplace\n\n        :param package_id: the package identification string\n        '
        if not self._package_manager.reinstallPackage(package_id):
            self.download(package_id, url, False)
        else:
            package = self.getPackageModel(package_id)
            if package:
                self.subscribeUserToPackage(package_id, str(package.sdk_version))

    def uninstallPackage(self, package_id: str) -> None:
        if False:
            print('Hello World!')
        'Uninstall a package from the Marketplace\n\n        :param package_id: the package identification string\n        '
        self._package_manager.removePackage(package_id)
        self.unsunscribeUserFromPackage(package_id)

    def updatePackage(self, package_id: str, url: str) -> None:
        if False:
            return 10
        'Update a package from the Marketplace\n\n        :param package_id: the package identification string\n        '
        self._package_manager.removePackage(package_id, force_add=not self._package_manager.isBundledPackage(package_id))
        self.download(package_id, url, True)