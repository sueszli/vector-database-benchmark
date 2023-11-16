import os
from typing import Optional, Dict, List, Callable, Any
from time import time
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import pyqtSlot, QUrl, pyqtSignal, pyqtProperty, QObject
from PyQt6.QtNetwork import QNetworkReply
from UM.FileHandler.FileHandler import FileHandler
from UM.Version import Version
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.Scene.SceneNode import SceneNode
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.NetworkedPrinterOutputDevice import AuthState
from cura.PrinterOutput.PrinterOutputDevice import ConnectionType
from .ClusterApiClient import ClusterApiClient
from .SendMaterialJob import SendMaterialJob
from ..ExportFileJob import ExportFileJob
from ..UltimakerNetworkedPrinterOutputDevice import UltimakerNetworkedPrinterOutputDevice
from ..Messages.PrintJobUploadBlockedMessage import PrintJobUploadBlockedMessage
from ..Messages.PrintJobUploadErrorMessage import PrintJobUploadErrorMessage
from ..Messages.PrintJobUploadSuccessMessage import PrintJobUploadSuccessMessage
from ..Models.Http.ClusterMaterial import ClusterMaterial
I18N_CATALOG = i18nCatalog('cura')

class LocalClusterOutputDevice(UltimakerNetworkedPrinterOutputDevice):
    activeCameraUrlChanged = pyqtSignal()
    CHECK_CLUSTER_INTERVAL = 10.0

    def __init__(self, device_id: str, address: str, properties: Dict[bytes, bytes], parent=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(device_id=device_id, address=address, properties=properties, connection_type=ConnectionType.NetworkConnection, parent=parent)
        self._timeout_time = 30
        self._cluster_api = None
        self._active_exported_job = None
        self._printer_select_dialog = None
        self.setAuthenticationState(AuthState.Authenticated)
        self._setInterfaceElements()
        self._active_camera_url = QUrl()

    def _setInterfaceElements(self) -> None:
        if False:
            while True:
                i = 10
        'Set all the interface elements and texts for this output device.'
        self.setPriority(3)
        self.setShortDescription(I18N_CATALOG.i18nc("@action:button Preceded by 'Ready to'.", 'Print over network'))
        self.setDescription(I18N_CATALOG.i18nc('@properties:tooltip', 'Print over network'))
        self.setConnectionText(I18N_CATALOG.i18nc('@info:status', 'Connected over the network'))

    def connect(self) -> None:
        if False:
            print('Hello World!')
        'Called when the connection to the cluster changes.'
        super().connect()
        self._update()
        self.sendMaterialProfiles()

    @pyqtProperty(QUrl, notify=activeCameraUrlChanged)
    def activeCameraUrl(self) -> QUrl:
        if False:
            i = 10
            return i + 15
        return self._active_camera_url

    @pyqtSlot(QUrl, name='setActiveCameraUrl')
    def setActiveCameraUrl(self, camera_url: QUrl) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self._active_camera_url != camera_url:
            self._active_camera_url = camera_url
            self.activeCameraUrlChanged.emit()

    @pyqtSlot(name='openPrintJobControlPanel')
    def openPrintJobControlPanel(self) -> None:
        if False:
            i = 10
            return i + 15
        QDesktopServices.openUrl(QUrl('http://' + self._address + '/print_jobs'))

    @pyqtSlot(name='openPrinterControlPanel')
    def openPrinterControlPanel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if Version(self.firmwareVersion) >= Version('7.0.2'):
            QDesktopServices.openUrl(QUrl('http://' + self._address + '/print_jobs'))
        else:
            QDesktopServices.openUrl(QUrl('http://' + self._address + '/printers'))

    @pyqtSlot(str, name='sendJobToTop')
    def sendJobToTop(self, print_job_uuid: str) -> None:
        if False:
            while True:
                i = 10
        self._getApiClient().movePrintJobToTop(print_job_uuid)

    @pyqtSlot(str, name='deleteJobFromQueue')
    def deleteJobFromQueue(self, print_job_uuid: str) -> None:
        if False:
            i = 10
            return i + 15
        self._getApiClient().deletePrintJob(print_job_uuid)

    @pyqtSlot(str, name='forceSendJob')
    def forceSendJob(self, print_job_uuid: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._getApiClient().forcePrintJob(print_job_uuid)

    def setJobState(self, print_job_uuid: str, action: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Set the remote print job state.\n\n        :param print_job_uuid: The UUID of the print job to set the state for.\n        :param action: The action to undertake ('pause', 'resume', 'abort').\n        "
        self._getApiClient().setPrintJobState(print_job_uuid, action)

    def _update(self) -> None:
        if False:
            while True:
                i = 10
        super()._update()
        if time() - self._time_of_last_request < self.CHECK_CLUSTER_INTERVAL:
            return
        self._getApiClient().getPrinters(self._updatePrinters)
        self._getApiClient().getPrintJobs(self._updatePrintJobs)
        self._updatePrintJobPreviewImages()

    def getMaterials(self, on_finished: Callable[[List[ClusterMaterial]], Any]) -> None:
        if False:
            while True:
                i = 10
        'Get a list of materials that are installed on the cluster host.'
        self._getApiClient().getMaterials(on_finished=on_finished)

    def sendMaterialProfiles(self) -> None:
        if False:
            print('Hello World!')
        'Sync the material profiles in Cura with the printer.\n\n        This gets called when connecting to a printer as well as when sending a print.\n        '
        job = SendMaterialJob(device=self)
        job.run()

    def requestWrite(self, nodes: List[SceneNode], file_name: Optional[str]=None, limit_mimetypes: bool=False, file_handler: Optional[FileHandler]=None, filter_by_machine: bool=False, **kwargs) -> None:
        if False:
            return 10
        'Send a print job to the cluster.'
        if self._progress.visible:
            PrintJobUploadBlockedMessage().show()
            return
        self.writeStarted.emit(self)
        job = ExportFileJob(file_handler=file_handler, nodes=nodes, firmware_version=self.firmwareVersion)
        job.finished.connect(self._onPrintJobCreated)
        job.start()

    @pyqtSlot(str, name='selectTargetPrinter')
    def selectTargetPrinter(self, unique_name: str='') -> None:
        if False:
            return 10
        'Allows the user to choose a printer to print with from the printer selection dialogue.\n\n        :param unique_name: The unique name of the printer to target.\n        '
        self._startPrintJobUpload(unique_name if unique_name != '' else None)

    def _onPrintJobCreated(self, job: ExportFileJob) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handler for when the print job was created locally.\n\n        It can now be sent over the network.\n        '
        self._active_exported_job = job
        if self.clusterSize > 1:
            self._showPrinterSelectionDialog()
            return
        self._startPrintJobUpload()

    def _showPrinterSelectionDialog(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Shows a dialog allowing the user to select which printer in a group to send a job to.'
        if not self._printer_select_dialog:
            plugin_path = CuraApplication.getInstance().getPluginRegistry().getPluginPath('UM3NetworkPrinting') or ''
            path = os.path.join(plugin_path, 'resources', 'qml', 'PrintWindow.qml')
            self._printer_select_dialog = CuraApplication.getInstance().createQmlComponent(path, {'OutputDevice': self})
        if self._printer_select_dialog is not None:
            self._printer_select_dialog.show()

    def _startPrintJobUpload(self, unique_name: str=None) -> None:
        if False:
            i = 10
            return i + 15
        'Upload the print job to the group.'
        if not self._active_exported_job:
            Logger.log('e', 'No active exported job to upload!')
            return
        self._progress.show()
        parts = [self._createFormPart('name=owner', bytes(self._getUserName(), 'utf-8'), 'text/plain'), self._createFormPart('name="file"; filename="%s"' % self._active_exported_job.getFileName(), self._active_exported_job.getOutput())]
        if unique_name is not None:
            parts.append(self._createFormPart('name=require_printer_name', bytes(unique_name, 'utf-8'), 'text/plain'))
        self.postFormWithParts('/cluster-api/v1/print_jobs/', parts, on_finished=self._onPrintUploadCompleted, on_progress=self._onPrintJobUploadProgress)
        self._active_exported_job = None

    def _onPrintJobUploadProgress(self, bytes_sent: int, bytes_total: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handler for print job upload progress.'
        percentage = bytes_sent / bytes_total if bytes_total else 0
        self._progress.setProgress(percentage * 100)
        self.writeProgress.emit()

    def _onPrintUploadCompleted(self, _: QNetworkReply) -> None:
        if False:
            i = 10
            return i + 15
        'Handler for when the print job was fully uploaded to the cluster.'
        self._progress.hide()
        PrintJobUploadSuccessMessage().show()
        self.writeFinished.emit()

    def _onUploadError(self, message: str=None) -> None:
        if False:
            print('Hello World!')
        'Displays the given message if uploading the mesh has failed\n\n        :param message: The message to display.\n        '
        self._progress.hide()
        PrintJobUploadErrorMessage(message).show()
        self.writeError.emit()

    def _updatePrintJobPreviewImages(self):
        if False:
            print('Hello World!')
        'Download all the images from the cluster and load their data in the print job models.'
        for print_job in self._print_jobs:
            if print_job.getPreviewImage() is None:
                self._getApiClient().getPrintJobPreviewImage(print_job.key, print_job.updatePreviewImageData)

    def _getApiClient(self) -> ClusterApiClient:
        if False:
            return 10
        'Get the API client instance.'
        if not self._cluster_api:
            self._cluster_api = ClusterApiClient(self.address, on_error=lambda error: Logger.log('e', str(error)))
        return self._cluster_api