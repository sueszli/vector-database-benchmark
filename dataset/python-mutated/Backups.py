from typing import Tuple, Optional, TYPE_CHECKING, Dict, Any
from cura.Backups.BackupsManager import BackupsManager
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class Backups:
    """The back-ups API provides a version-proof bridge between Cura's

    BackupManager and plug-ins that hook into it.

    Usage:

    .. code-block:: python

       from cura.API import CuraAPI
       api = CuraAPI()
       api.backups.createBackup()
       api.backups.restoreBackup(my_zip_file, {"cura_release": "3.1"})
    """

    def __init__(self, application: 'CuraApplication') -> None:
        if False:
            return 10
        self.manager = BackupsManager(application)

    def createBackup(self) -> Tuple[Optional[bytes], Optional[Dict[str, Any]]]:
        if False:
            for i in range(10):
                print('nop')
        'Create a new back-up using the BackupsManager.\n\n        :return: Tuple containing a ZIP file with the back-up data and a dict with metadata about the back-up.\n        '
        return self.manager.createBackup()

    def restoreBackup(self, zip_file: bytes, meta_data: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Restore a back-up using the BackupsManager.\n\n        :param zip_file: A ZIP file containing the actual back-up data.\n        :param meta_data: Some metadata needed for restoring a back-up, like the Cura version number.\n        '
        return self.manager.restoreBackup(zip_file, meta_data)