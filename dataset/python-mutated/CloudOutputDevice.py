from time import time
import os
from typing import cast, List, Optional
from PyQt6.QtCore import QObject, QUrl, pyqtProperty, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkReply, QNetworkRequest
from UM import i18nCatalog
from UM.FileHandler.FileHandler import FileHandler
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode
from UM.Version import Version
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.NetworkedPrinterOutputDevice import AuthState
from cura.PrinterOutput.PrinterOutputDevice import ConnectionType
from cura.Scene.GCodeListDecorator import GCodeListDecorator
from cura.Scene.SliceableObjectDecorator import SliceableObjectDecorator
from .CloudApiClient import CloudApiClient
from ..ExportFileJob import ExportFileJob
from ..Messages.PrintJobAwaitingApprovalMessage import PrintJobPendingApprovalMessage
from ..UltimakerNetworkedPrinterOutputDevice import UltimakerNetworkedPrinterOutputDevice
from ..Messages.PrintJobUploadBlockedMessage import PrintJobUploadBlockedMessage
from ..Messages.PrintJobUploadErrorMessage import PrintJobUploadErrorMessage
from ..Messages.PrintJobUploadQueueFullMessage import PrintJobUploadQueueFullMessage
from ..Messages.PrintJobUploadSuccessMessage import PrintJobUploadSuccessMessage
from ..Models.Http.CloudClusterResponse import CloudClusterResponse
from ..Models.Http.CloudClusterStatus import CloudClusterStatus
from ..Models.Http.CloudPrintJobUploadRequest import CloudPrintJobUploadRequest
from ..Models.Http.CloudPrintResponse import CloudPrintResponse
from ..Models.Http.CloudPrintJobResponse import CloudPrintJobResponse
from ..Models.Http.ClusterPrinterStatus import ClusterPrinterStatus
from ..Models.Http.ClusterPrintJobStatus import ClusterPrintJobStatus
I18N_CATALOG = i18nCatalog('cura')

class CloudOutputDevice(UltimakerNetworkedPrinterOutputDevice):
    """The cloud output device is a network output device that works remotely but has limited functionality.

    Currently, it only supports viewing the printer and print job status and adding a new job to the queue.
    As such, those methods have been implemented here.
    Note that this device represents a single remote cluster, not a list of multiple clusters.
    """
    CHECK_CLUSTER_INTERVAL = 10.0
    NETWORK_RESPONSE_CONSIDER_OFFLINE = 15.0
    PRINT_JOB_ACTIONS_MIN_VERSION = Version('5.2.12')
    PRINT_JOB_ACTIONS_MIN_VERSION_METHOD = Version('2.700')
    _cloudClusterPrintersChanged = pyqtSignal()

    def __init__(self, api_client: CloudApiClient, cluster: CloudClusterResponse, parent: QObject=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a new cloud output device\n\n        :param api_client: The client that will run the API calls\n        :param cluster: The device response received from the cloud API.\n        :param parent: The optional parent of this output device.\n        '
        properties = {b'address': cluster.host_internal_ip.encode() if cluster.host_internal_ip else b'', b'name': cluster.friendly_name.encode() if cluster.friendly_name else b'', b'firmware_version': cluster.host_version.encode() if cluster.host_version else b'', b'printer_type': cluster.printer_type.encode() if cluster.printer_type else b'', b'cluster_size': str(cluster.printer_count).encode() if cluster.printer_count else b'1'}
        super().__init__(device_id=cluster.cluster_id, address='', connection_type=ConnectionType.CloudConnection, properties=properties, parent=parent)
        self._api = api_client
        self._account = api_client.account
        self._cluster = cluster
        self.setAuthenticationState(AuthState.NotAuthenticated)
        self._setInterfaceElements()
        self.printersChanged.connect(self._cloudClusterPrintersChanged)
        self._account.permissionsChanged.connect(self.permissionsChanged)
        self._received_printers = None
        self._received_print_jobs = None
        self._tool_path = None
        self._pre_upload_print_job = None
        self._uploaded_print_job = None
        CuraApplication.getInstance().getBackend().backendDone.connect(self._resetPrintJob)
        CuraApplication.getInstance().getController().getScene().sceneChanged.connect(self._onSceneChanged)

    def connect(self) -> None:
        if False:
            return 10
        'Connects this device.'
        if self.isConnected():
            return
        Logger.log('i', 'Attempting to connect to cluster %s', self.key)
        super().connect()
        self._update()

    def disconnect(self) -> None:
        if False:
            i = 10
            return i + 15
        'Disconnects the device'
        if not self.isConnected():
            return
        super().disconnect()
        Logger.log('i', 'Disconnected from cluster %s', self.key)

    def _onSceneChanged(self, node: SceneNode):
        if False:
            return 10
        if node.getDecorator(GCodeListDecorator) or node.getDecorator(SliceableObjectDecorator):
            self._resetPrintJob()

    def _resetPrintJob(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Resets the print job that was uploaded to force a new upload, runs whenever slice finishes.'
        self._tool_path = None
        self._pre_upload_print_job = None
        self._uploaded_print_job = None

    def matchesNetworkKey(self, network_key: str) -> bool:
        if False:
            while True:
                i = 10
        "Checks whether the given network key is found in the cloud's host name"
        if network_key.startswith(str(self.clusterData.host_name or '')):
            return True
        if network_key.endswith(str(self.clusterData.host_internal_ip or '')):
            return True
        return False

    def _setInterfaceElements(self) -> None:
        if False:
            print('Hello World!')
        'Set all the interface elements and texts for this output device.'
        self.setPriority(2)
        self.setShortDescription(I18N_CATALOG.i18nc('@action:button', 'Print via cloud'))
        self.setDescription(I18N_CATALOG.i18nc('@properties:tooltip', 'Print via cloud'))
        self.setConnectionText(I18N_CATALOG.i18nc('@info:status', 'Connected via cloud'))

    def _update(self) -> None:
        if False:
            return 10
        'Called when the network data should be updated.'
        super()._update()
        if time() - self._time_of_last_request < self.CHECK_CLUSTER_INTERVAL:
            return
        self._time_of_last_request = time()
        if self._account.isLoggedIn:
            self.setAuthenticationState(AuthState.Authenticated)
            self._last_request_time = time()
            self._api.getClusterStatus(self.key, self._onStatusCallFinished)
        else:
            self.setAuthenticationState(AuthState.NotAuthenticated)

    def _onStatusCallFinished(self, status: CloudClusterStatus) -> None:
        if False:
            return 10
        'Method called when HTTP request to status endpoint is finished.\n\n        Contains both printers and print jobs statuses in a single response.\n        '
        self._responseReceived()
        if status.printers != self._received_printers:
            self._received_printers = status.printers
            self._updatePrinters(status.printers)
        if status.print_jobs != self._received_print_jobs:
            self._received_print_jobs = status.print_jobs
            self._updatePrintJobs(status.print_jobs)

    def requestWrite(self, nodes: List[SceneNode], file_name: Optional[str]=None, limit_mimetypes: bool=False, file_handler: Optional[FileHandler]=None, filter_by_machine: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Called when Cura requests an output device to receive a (G-code) file.'
        if self._progress.visible:
            PrintJobUploadBlockedMessage().show()
            return
        self._progress.show()
        self.writeStarted.emit(self)
        if self._uploaded_print_job:
            Logger.log('i', 'Current mesh is already attached to a print-job, immediately request reprint.')
            self._api.requestPrint(self.key, self._uploaded_print_job.job_id, self._onPrintUploadCompleted, self._onPrintUploadSpecificError)
            return
        job = ExportFileJob(file_handler=file_handler, nodes=nodes, firmware_version=self.firmwareVersion)
        job.finished.connect(self._onPrintJobCreated)
        job.start()

    def _onPrintJobCreated(self, job: ExportFileJob) -> None:
        if False:
            print('Hello World!')
        'Handler for when the print job was created locally.\n\n        It can now be sent over the cloud.\n        '
        output = job.getOutput()
        self._tool_path = output
        file_name = job.getFileName()
        request = CloudPrintJobUploadRequest(job_name=os.path.splitext(file_name)[0], file_size=len(output), content_type=job.getMimeType())
        self._api.requestUpload(request, self._uploadPrintJob)

    def _uploadPrintJob(self, job_response: CloudPrintJobResponse) -> None:
        if False:
            return 10
        'Uploads the mesh when the print job was registered with the cloud API.\n\n        :param job_response: The response received from the cloud API.\n        '
        if not self._tool_path:
            return self._onUploadError()
        self._pre_upload_print_job = job_response
        self._api.uploadToolPath(job_response, self._tool_path, self._onPrintJobUploaded, self._progress.update, self._onUploadError)

    def _onPrintJobUploaded(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Requests the print to be sent to the printer when we finished uploading\n        the mesh.\n        '
        self._progress.update(100)
        print_job = cast(CloudPrintJobResponse, self._pre_upload_print_job)
        if not print_job:
            self._pre_upload_print_job = None
            self._uploaded_print_job = None
            Logger.log('w', 'Interference from another job uploaded at roughly the same time, not uploading print!')
            return
        self._api.requestPrint(self.key, print_job.job_id, self._onPrintUploadCompleted, self._onPrintUploadSpecificError)

    def _onPrintUploadCompleted(self, response: CloudPrintResponse) -> None:
        if False:
            print('Hello World!')
        'Shows a message when the upload has succeeded\n\n        :param response: The response from the cloud API.\n        '
        self._uploaded_print_job = self._pre_upload_print_job
        self._progress.hide()
        if response:
            message = PrintJobUploadSuccessMessage()
            message.addAction('monitor print', name=I18N_CATALOG.i18nc('@action:button', 'Monitor print'), icon='', description=I18N_CATALOG.i18nc('@action:tooltip', 'Track the print in Ultimaker Digital Factory'), button_align=message.ActionButtonAlignment.ALIGN_RIGHT)
            df_url = f'https://digitalfactory.ultimaker.com/app/jobs/{self._cluster.cluster_id}?utm_source=cura&utm_medium=software&utm_campaign=message-printjob-sent'
            message.pyQtActionTriggered.connect(lambda message, action: (QDesktopServices.openUrl(QUrl(df_url)), message.hide()))
            message.show()
        else:
            PrintJobPendingApprovalMessage(self._cluster.cluster_id).show()
        self.writeFinished.emit()

    def _onPrintUploadSpecificError(self, reply: 'QNetworkReply', _: 'QNetworkReply.NetworkError'):
        if False:
            while True:
                i = 10
        '\n        Displays a message when an error occurs specific to uploading print job (i.e. queue is full).\n        '
        error_code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if error_code == 409:
            PrintJobUploadQueueFullMessage().show()
        else:
            PrintJobUploadErrorMessage(I18N_CATALOG.i18nc('@error:send', 'Unknown error code when uploading print job: {0}', error_code)).show()
        Logger.log('w', 'Upload of print job failed specifically with error code {}'.format(error_code))
        self._progress.hide()
        self._pre_upload_print_job = None
        self._uploaded_print_job = None
        self.writeError.emit()

    def _onUploadError(self, message: str=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Displays the given message if uploading the mesh has failed due to a generic error (i.e. lost connection).\n        :param message: The message to display.\n        '
        Logger.log('w', 'Upload error with message {}'.format(message))
        self._progress.hide()
        self._pre_upload_print_job = None
        self._uploaded_print_job = None
        PrintJobUploadErrorMessage(message).show()
        self.writeError.emit()

    @pyqtProperty(bool, notify=_cloudClusterPrintersChanged)
    def supportsPrintJobActions(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the printer that this output device represents supports print job actions via the cloud.'
        if not self._printers:
            return False
        version_number = self.printers[0].firmwareVersion.split('.')
        if len(version_number) > 2:
            firmware_version = Version([version_number[0], version_number[1], version_number[2]])
            return firmware_version >= self.PRINT_JOB_ACTIONS_MIN_VERSION
        else:
            firmware_version = Version([version_number[0], version_number[1]])
            return firmware_version >= self.PRINT_JOB_ACTIONS_MIN_VERSION_METHOD

    @pyqtProperty(bool, constant=True)
    def supportsPrintJobQueue(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Gets whether the printer supports a queue'
        return 'queue' in self._cluster.capabilities if self._cluster.capabilities else True

    def setJobState(self, print_job_uuid: str, state: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the remote print job state.'
        self._api.doPrintJobAction(self._cluster.cluster_id, print_job_uuid, state)

    @pyqtSlot(str, name='sendJobToTop')
    def sendJobToTop(self, print_job_uuid: str) -> None:
        if False:
            print('Hello World!')
        self._api.doPrintJobAction(self._cluster.cluster_id, print_job_uuid, 'move', {'list': 'queued', 'to_position': 0})

    @pyqtSlot(str, name='deleteJobFromQueue')
    def deleteJobFromQueue(self, print_job_uuid: str) -> None:
        if False:
            while True:
                i = 10
        self._api.doPrintJobAction(self._cluster.cluster_id, print_job_uuid, 'remove')

    @pyqtSlot(str, name='forceSendJob')
    def forceSendJob(self, print_job_uuid: str) -> None:
        if False:
            i = 10
            return i + 15
        self._api.doPrintJobAction(self._cluster.cluster_id, print_job_uuid, 'force')

    @pyqtSlot(name='openPrintJobControlPanel')
    def openPrintJobControlPanel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        QDesktopServices.openUrl(QUrl(f'{self.clusterCloudUrl}?utm_source=cura&utm_medium=software&utm_campaign=monitor-manage-browser'))

    @pyqtSlot(name='openPrinterControlPanel')
    def openPrinterControlPanel(self) -> None:
        if False:
            i = 10
            return i + 15
        QDesktopServices.openUrl(QUrl(f'{self.clusterCloudUrl}?utm_source=cura&utm_medium=software&utm_campaign=monitor-manage-printer'))
    permissionsChanged = pyqtSignal()

    @pyqtProperty(bool, notify=permissionsChanged)
    def canReadPrintJobs(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether this user can read the list of print jobs and their properties.\n        '
        return 'digital-factory.print-job.read' in self._account.permissions

    @pyqtProperty(bool, notify=permissionsChanged)
    def canWriteOthersPrintJobs(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether this user can change things about print jobs made by other\n        people.\n        '
        return 'digital-factory.print-job.write' in self._account.permissions

    @pyqtProperty(bool, notify=permissionsChanged)
    def canWriteOwnPrintJobs(self) -> bool:
        if False:
            return 10
        '\n        Whether this user can change things about print jobs made by them.\n        '
        return 'digital-factory.print-job.write.own' in self._account.permissions

    @pyqtProperty(bool, constant=True)
    def canReadPrinterDetails(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether this user can read the status of the printer.\n        '
        return 'digital-factory.printer.read' in self._account.permissions

    @property
    def clusterData(self) -> CloudClusterResponse:
        if False:
            return 10
        'Gets the cluster response from which this device was created.'
        return self._cluster

    @clusterData.setter
    def clusterData(self, value: CloudClusterResponse) -> None:
        if False:
            return 10
        'Updates the cluster data from the cloud.'
        self._cluster = value

    @property
    def clusterCloudUrl(self) -> str:
        if False:
            return 10
        'Gets the URL on which to monitor the cluster via the cloud.'
        root_url_prefix = '-staging' if self._account.is_staging else ''
        return f'https://digitalfactory{root_url_prefix}.ultimaker.com/app/jobs/{self.clusterData.cluster_id}'

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        CuraApplication.getInstance().getBackend().backendDone.disconnect(self._resetPrintJob)
        CuraApplication.getInstance().getController().getScene().sceneChanged.disconnect(self._onSceneChanged)