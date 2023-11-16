import base64
import hashlib
import threading
from tempfile import NamedTemporaryFile
from typing import Optional, Any, Dict
from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest
from UM.Job import Job
from UM.Logger import Logger
from UM.PackageManager import catalog
from UM.TaskManagement.HttpRequestManager import HttpRequestManager
from cura.CuraApplication import CuraApplication

class RestoreBackupJob(Job):
    """Downloads a backup and overwrites local configuration with the backup.

     When `Job.finished` emits, `restore_backup_error_message` will either be `""` (no error) or an error message
     """
    DISK_WRITE_BUFFER_SIZE = 512 * 1024
    DEFAULT_ERROR_MESSAGE = catalog.i18nc('@info:backup_status', 'There was an error trying to restore your backup.')

    def __init__(self, backup: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        ' Create a new restore Job. start the job by calling start()\n\n        :param backup: A dict containing a backup spec\n        '
        super().__init__()
        self._job_done = threading.Event()
        self._backup = backup
        self.restore_backup_error_message = ''

    def run(self) -> None:
        if False:
            print('Hello World!')
        url = self._backup.get('download_url')
        assert url is not None
        HttpRequestManager.getInstance().get(url=url, callback=self._onRestoreRequestCompleted, error_callback=self._onRestoreRequestCompleted)
        self._job_done.wait()

    def _onRestoreRequestCompleted(self, reply: QNetworkReply, error: Optional['QNetworkReply.NetworkError']=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not HttpRequestManager.replyIndicatesSuccess(reply, error):
            Logger.warning('Requesting backup failed, response code %s while trying to connect to %s', reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute), reply.url())
            self.restore_backup_error_message = self.DEFAULT_ERROR_MESSAGE
            self._job_done.set()
            return
        try:
            temporary_backup_file = NamedTemporaryFile(delete=False)
            with open(temporary_backup_file.name, 'wb') as write_backup:
                app = CuraApplication.getInstance()
                bytes_read = reply.read(self.DISK_WRITE_BUFFER_SIZE)
                while bytes_read:
                    write_backup.write(bytes_read)
                    bytes_read = reply.read(self.DISK_WRITE_BUFFER_SIZE)
                    app.processEvents()
        except EnvironmentError as e:
            Logger.log('e', f'Unable to save backed up files due to computer limitations: {str(e)}')
            self.restore_backup_error_message = self.DEFAULT_ERROR_MESSAGE
            self._job_done.set()
            return
        if not self._verifyMd5Hash(temporary_backup_file.name, self._backup.get('md5_hash', '')):
            Logger.log('w', 'Remote and local MD5 hashes do not match, not restoring backup.')
            self.restore_backup_error_message = self.DEFAULT_ERROR_MESSAGE
        with open(temporary_backup_file.name, 'rb') as read_backup:
            cura_api = CuraApplication.getInstance().getCuraAPI()
            cura_api.backups.restoreBackup(read_backup.read(), self._backup.get('metadata', {}))
        self._job_done.set()

    @staticmethod
    def _verifyMd5Hash(file_path: str, known_hash: str) -> bool:
        if False:
            return 10
        'Verify the MD5 hash of a file.\n\n        :param file_path: Full path to the file.\n        :param known_hash: The known MD5 hash of the file.\n        :return: Success or not.\n        '
        with open(file_path, 'rb') as read_backup:
            local_md5_hash = base64.b64encode(hashlib.md5(read_backup.read()).digest(), altchars=b'_-').decode('utf-8')
            return known_hash == local_md5_hash