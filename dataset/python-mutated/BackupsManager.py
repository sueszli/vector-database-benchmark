from typing import Dict, Optional, Tuple, TYPE_CHECKING
from UM.Logger import Logger
from UM.Version import Version
from cura.Backups.Backup import Backup
if TYPE_CHECKING:
    from cura.CuraApplication import CuraApplication

class BackupsManager:
    """
    The BackupsManager is responsible for managing the creating and restoring of
    back-ups.

    Back-ups themselves are represented in a different class.
    """

    def __init__(self, application: 'CuraApplication') -> None:
        if False:
            i = 10
            return i + 15
        self._application = application

    def createBackup(self) -> Tuple[Optional[bytes], Optional[Dict[str, str]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a back-up of the current configuration.\n\n        :return: A tuple containing a ZipFile (the actual back-up) and a dict containing some metadata (like version).\n        '
        self._disableAutoSave()
        backup = Backup(self._application)
        backup.makeFromCurrent()
        self._enableAutoSave()
        return (backup.zip_file, backup.meta_data)

    def restoreBackup(self, zip_file: bytes, meta_data: Dict[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore a back-up from a given ZipFile.\n\n        :param zip_file: A bytes object containing the actual back-up.\n        :param meta_data: A dict containing some metadata that is needed to restore the back-up correctly.\n        '
        if not meta_data.get('cura_release', None):
            Logger.log('w', 'Tried to restore a backup without specifying a Cura version number.')
            return
        self._disableAutoSave()
        backup = Backup(self._application, zip_file=zip_file, meta_data=meta_data)
        restored = backup.restore()
        if restored:
            self._application.windowClosed(save_data=False)

    def _disableAutoSave(self) -> None:
        if False:
            print('Hello World!')
        'Here we (try to) disable the saving as it might interfere with restoring a back-up.'
        self._application.enableSave(False)
        auto_save = self._application.getAutoSave()
        if auto_save:
            auto_save.setEnabled(False)
        else:
            Logger.log('e', 'Unable to disable the autosave as application init has not been completed')

    def _enableAutoSave(self) -> None:
        if False:
            return 10
        "Re-enable auto-save and other saving after we're done."
        self._application.enableSave(True)
        auto_save = self._application.getAutoSave()
        if auto_save:
            auto_save.setEnabled(True)
        else:
            Logger.log('e', 'Unable to enable the autosave as application init has not been completed')