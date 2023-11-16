from typing import Optional, TYPE_CHECKING, Dict, List
from .Constants import PACKAGES_URL
from .PackageModel import PackageModel
from .RemotePackageList import RemotePackageList
from PyQt6.QtCore import pyqtSignal, QObject, pyqtProperty, QCoreApplication
from UM.TaskManagement.HttpRequestManager import HttpRequestManager
from UM.i18n import i18nCatalog
if TYPE_CHECKING:
    from PyQt6.QtCore import QObject, pyqtProperty, pyqtSignal
catalog = i18nCatalog('cura')

class MissingPackageList(RemotePackageList):

    def __init__(self, packages_metadata: List[Dict[str, str]], parent: Optional['QObject']=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._packages_metadata: List[Dict[str, str]] = packages_metadata
        self._search_type = 'package_ids'
        self._requested_search_string = ','.join(map(lambda package: package['id'], packages_metadata))

    def _parseResponse(self, reply: 'QNetworkReply') -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._parseResponse(reply)
        if not self.hasMore:
            self._addPackagesMissingFromRequest()

    def _addPackagesMissingFromRequest(self) -> None:
        if False:
            print('Hello World!')
        'Create cards for packages the user needs to install that could not be found'
        returned_packages_ids = [item['package'].packageId for item in self._items]
        for package_metadata in self._packages_metadata:
            if package_metadata['id'] not in returned_packages_ids:
                package_type = package_metadata['type'] if 'type' in package_metadata else 'material'
                package = PackageModel.fromIncompletePackageInformation(package_metadata['display_name'], package_metadata['package_version'], package_type)
                self.appendItem({'package': package})
        self.itemsChanged.emit()