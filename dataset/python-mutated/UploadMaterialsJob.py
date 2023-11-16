import enum
import functools
import json
import os
from PyQt6.QtCore import QUrl
import tempfile
import cura.CuraApplication
from cura.Settings.CuraContainerRegistry import CuraContainerRegistry
from cura.UltimakerCloud import UltimakerCloudConstants
from cura.UltimakerCloud.UltimakerCloudScope import UltimakerCloudScope
from UM.i18n import i18nCatalog
from UM.Job import Job
from UM.Logger import Logger
from UM.Signal import Signal
from UM.TaskManagement.HttpRequestManager import HttpRequestManager
from UM.TaskManagement.HttpRequestScope import JsonDecoratorScope
from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from PyQt6.QtNetwork import QNetworkReply
    from cura.UltimakerCloud.CloudMaterialSync import CloudMaterialSync
catalog = i18nCatalog('cura')

class UploadMaterialsError(Exception):
    """
    Class to indicate something went wrong while uploading.
    """
    pass

class UploadMaterialsJob(Job):
    """
    Job that uploads a set of materials to the Digital Factory.

    The job has a number of stages:
    - First, it generates an archive of all materials. This typically takes a lot of processing power during which the
      GIL remains locked.
    - Then it requests the API to upload an archive.
    - Then it uploads the archive to the URL given by the first request.
    - Then it tells the API that the archive can be distributed to the printers.
    """
    UPLOAD_REQUEST_URL = f'{UltimakerCloudConstants.CuraCloudAPIRoot}/connect/v1/materials/upload'
    UPLOAD_CONFIRM_URL = UltimakerCloudConstants.CuraCloudAPIRoot + '/connect/v1/clusters/{cluster_id}/printers/{cluster_printer_id}/action/import_material'

    class Result(enum.IntEnum):
        SUCCESS = 0
        FAILED = 1

    class PrinterStatus(enum.Enum):
        UPLOADING = 'uploading'
        SUCCESS = 'success'
        FAILED = 'failed'

    def __init__(self, material_sync: 'CloudMaterialSync'):
        if False:
            while True:
                i = 10
        super().__init__()
        self._material_sync = material_sync
        self._scope = JsonDecoratorScope(UltimakerCloudScope(cura.CuraApplication.CuraApplication.getInstance()))
        self._archive_filename = None
        self._archive_remote_id = None
        self._printer_sync_status = {}
        self._printer_metadata = []
        self.processProgressChanged.connect(self._onProcessProgressChanged)
    uploadCompleted = Signal()
    processProgressChanged = Signal()
    uploadProgressChanged = Signal()

    def run(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates an archive of materials and starts uploading that archive to the cloud.\n        '
        self._printer_metadata = CuraContainerRegistry.getInstance().findContainerStacksMetadata(type='machine', connection_type='3', is_online='True', host_guid='*', um_cloud_cluster_id='*')
        self._printer_metadata = [printer_data for printer_data in self._printer_metadata if UltimakerCloudConstants.META_CAPABILITIES in printer_data and 'import_material' in printer_data[UltimakerCloudConstants.META_CAPABILITIES]]
        for printer in self._printer_metadata:
            self._printer_sync_status[printer['host_guid']] = self.PrinterStatus.UPLOADING.value
        try:
            archive_file = tempfile.NamedTemporaryFile('wb', delete=False)
            archive_file.close()
            self._archive_filename = archive_file.name
            self._material_sync.exportAll(QUrl.fromLocalFile(self._archive_filename), notify_progress=self.processProgressChanged)
        except OSError as e:
            Logger.error(f'Failed to create archive of materials to sync with printers: {type(e)} - {e}')
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'Failed to create archive of materials to sync with printers.')))
            return
        try:
            file_size = os.path.getsize(self._archive_filename)
        except OSError as e:
            Logger.error(f'Failed to load the archive of materials to sync it with printers: {type(e)} - {e}')
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'Failed to load the archive of materials to sync it with printers.')))
            return
        request_metadata = {'data': {'file_size': file_size, 'material_profile_name': 'cura.umm', 'content_type': 'application/zip', 'origin': 'cura'}}
        request_payload = json.dumps(request_metadata).encode('UTF-8')
        http = HttpRequestManager.getInstance()
        http.put(url=self.UPLOAD_REQUEST_URL, data=request_payload, callback=self.onUploadRequestCompleted, error_callback=self.onError, scope=self._scope)

    def onUploadRequestCompleted(self, reply: 'QNetworkReply') -> None:
        if False:
            return 10
        '\n        Triggered when we successfully requested to upload a material archive.\n\n        We then need to start uploading the material archive to the URL that the request answered with.\n        :param reply: The reply from the server to our request to upload an archive.\n        '
        response_data = HttpRequestManager.readJSON(reply)
        if response_data is None:
            Logger.error(f'Invalid response to material upload request. Could not parse JSON data.')
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'The response from Digital Factory appears to be corrupted.')))
            return
        if 'data' not in response_data:
            Logger.error(f"Invalid response to material upload request: Missing 'data' field that contains the entire response.")
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'The response from Digital Factory is missing important information.')))
            return
        if 'upload_url' not in response_data['data']:
            Logger.error(f"Invalid response to material upload request: Missing 'upload_url' field to upload archive to.")
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'The response from Digital Factory is missing important information.')))
            return
        if 'material_profile_id' not in response_data['data']:
            Logger.error(f"Invalid response to material upload request: Missing 'material_profile_id' to communicate about the materials with the server.")
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'The response from Digital Factory is missing important information.')))
            return
        upload_url = response_data['data']['upload_url']
        self._archive_remote_id = response_data['data']['material_profile_id']
        try:
            with open(cast(str, self._archive_filename), 'rb') as f:
                file_data = f.read()
        except OSError as e:
            Logger.error(f'Failed to load archive back in for sending to cloud: {type(e)} - {e}')
            self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'Failed to load the archive of materials to sync it with printers.')))
            return
        http = HttpRequestManager.getInstance()
        http.put(url=upload_url, data=file_data, callback=self.onUploadCompleted, error_callback=self.onError, scope=self._scope)

    def onUploadCompleted(self, reply: 'QNetworkReply') -> None:
        if False:
            return 10
        "\n        When we've successfully uploaded the archive to the cloud, we need to notify the API to start syncing that\n        archive to every printer.\n        :param reply: The reply from the cloud storage when the upload succeeded.\n        "
        for container_stack in self._printer_metadata:
            cluster_id = container_stack['um_cloud_cluster_id']
            printer_id = container_stack['host_guid']
            http = HttpRequestManager.getInstance()
            http.post(url=self.UPLOAD_CONFIRM_URL.format(cluster_id=cluster_id, cluster_printer_id=printer_id), callback=functools.partial(self.onUploadConfirmed, printer_id), error_callback=functools.partial(self.onUploadConfirmed, printer_id), scope=self._scope, data=json.dumps({'data': {'material_profile_id': self._archive_remote_id}}).encode('UTF-8'))

    def onUploadConfirmed(self, printer_id: str, reply: 'QNetworkReply', error: Optional['QNetworkReply.NetworkError']=None) -> None:
        if False:
            print('Hello World!')
        '\n        Triggered when we\'ve got a confirmation that the material is synced with the printer, or that syncing failed.\n\n        If syncing succeeded we mark this printer as having the status "success". If it failed we mark the printer as\n        "failed". If this is the last upload that needed to be completed, we complete the job with either a success\n        state (every printer successfully synced) or a failed state (any printer failed).\n        :param printer_id: The printer host_guid that we completed syncing with.\n        :param reply: The reply that the server gave to confirm.\n        :param error: If the request failed, this error gives an indication what happened.\n        '
        if error is not None:
            Logger.error(f'Failed to confirm uploading material archive to printer {printer_id}: {error}')
            self._printer_sync_status[printer_id] = self.PrinterStatus.FAILED.value
        else:
            self._printer_sync_status[printer_id] = self.PrinterStatus.SUCCESS.value
        still_uploading = len([val for val in self._printer_sync_status.values() if val == self.PrinterStatus.UPLOADING.value])
        self.uploadProgressChanged.emit(0.8 + (len(self._printer_sync_status) - still_uploading) / len(self._printer_sync_status), self.getPrinterSyncStatus())
        if still_uploading == 0:
            if self.PrinterStatus.FAILED.value in self._printer_sync_status.values():
                self.setResult(self.Result.FAILED)
                self.setError(UploadMaterialsError(catalog.i18nc('@text:error', 'Failed to connect to Digital Factory to sync materials with some of the printers.')))
            else:
                self.setResult(self.Result.SUCCESS)
            self.uploadCompleted.emit(self.getResult(), self.getError())

    def onError(self, reply: 'QNetworkReply', error: Optional['QNetworkReply.NetworkError']) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Used as callback from HTTP requests when the request failed.\n\n        The given network error from the `HttpRequestManager` is logged, and the job is marked as failed.\n        :param reply: The main reply of the server. This reply will most likely not be valid.\n        :param error: The network error (Qt's enum) that occurred.\n        "
        Logger.error(f'Failed to upload material archive: {error}')
        self.failed(UploadMaterialsError(catalog.i18nc('@text:error', 'Failed to connect to Digital Factory.')))

    def getPrinterSyncStatus(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        For each printer, identified by host_guid, this gives the current status of uploading the material archive.\n\n        The possible states are given in the PrinterStatus enum.\n        :return: A dictionary with printer host_guids as keys, and their status as values.\n        '
        return self._printer_sync_status

    def failed(self, error: UploadMaterialsError) -> None:
        if False:
            while True:
                i = 10
        '\n        Helper function for when we have a general failure.\n\n        This sets the sync status for all printers to failed, sets the error on\n        the job and the result of the job to FAILED.\n        :param error: An error to show to the user.\n        '
        self.setResult(self.Result.FAILED)
        self.setError(error)
        for printer_id in self._printer_sync_status:
            self._printer_sync_status[printer_id] = self.PrinterStatus.FAILED.value
        self.uploadProgressChanged.emit(1.0, self.getPrinterSyncStatus())
        self.uploadCompleted.emit(self.getResult(), self.getError())

    def _onProcessProgressChanged(self, progress: float) -> None:
        if False:
            while True:
                i = 10
        '\n        When we progress in the process of uploading materials, we not only signal the new progress (float from 0 to 1)\n        but we also signal the current status of every printer. These are emitted as the two parameters of the signal.\n        :param progress: The progress of this job, between 0 and 1.\n        '
        self.uploadProgressChanged.emit(progress * 0.8, self.getPrinterSyncStatus())