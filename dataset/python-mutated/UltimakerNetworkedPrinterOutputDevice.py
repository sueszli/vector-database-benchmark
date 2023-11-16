import os
from time import time
from typing import List, Optional, Dict
from PyQt6.QtCore import pyqtProperty, pyqtSignal, QObject, pyqtSlot, QUrl
from UM.Logger import Logger
from UM.Qt.Duration import Duration, DurationFormat
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.Models.PrinterOutputModel import PrinterOutputModel
from cura.PrinterOutput.NetworkedPrinterOutputDevice import NetworkedPrinterOutputDevice, AuthState
from cura.PrinterOutput.PrinterOutputDevice import ConnectionType, ConnectionState
from .Utils import formatTimeCompleted, formatDateCompleted
from .ClusterOutputController import ClusterOutputController
from .Messages.PrintJobUploadProgressMessage import PrintJobUploadProgressMessage
from .Messages.NotClusterHostMessage import NotClusterHostMessage
from .Models.UM3PrintJobOutputModel import UM3PrintJobOutputModel
from .Models.Http.ClusterPrinterStatus import ClusterPrinterStatus
from .Models.Http.ClusterPrintJobStatus import ClusterPrintJobStatus

class UltimakerNetworkedPrinterOutputDevice(NetworkedPrinterOutputDevice):
    """Output device class that forms the basis of Ultimaker networked printer output devices.

    Currently used for local networking and cloud printing using Ultimaker Connect.
    This base class primarily contains all the Qt properties and slots needed for the monitor page to work.
    """
    META_NETWORK_KEY = 'um_network_key'
    META_CLUSTER_ID = 'um_cloud_cluster_id'
    printJobsChanged = pyqtSignal()
    activePrinterChanged = pyqtSignal()
    _clusterPrintersChanged = pyqtSignal()
    QUEUED_PRINT_JOBS_STATES = {'queued', 'error'}

    def __init__(self, device_id: str, address: str, properties: Dict[bytes, bytes], connection_type: ConnectionType, parent=None) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(device_id=device_id, address=address, properties=properties, connection_type=connection_type, parent=parent)
        self.printersChanged.connect(self._clusterPrintersChanged)
        self._time_of_last_response = time()
        self._time_of_last_request = time()
        self.setName(self.getProperty('name'))
        definitions = CuraApplication.getInstance().getContainerRegistry().findContainers(id=self.printerType)
        self._printer_type_name = definitions[0].getName() if definitions else ''
        self._printers = []
        self._has_received_printers = False
        self._print_jobs = []
        self._active_printer = None
        self._authentication_state = AuthState.NotAuthenticated
        self._loadMonitorTab()
        self._progress = PrintJobUploadProgressMessage()
        self._timeout_time = 30
        self._num_is_host_check_failed = 0

    @pyqtProperty(str, constant=True)
    def address(self) -> str:
        if False:
            while True:
                i = 10
        'The IP address of the printer.'
        return self._address

    @pyqtProperty(str, constant=True)
    def printerTypeName(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'The display name of the printer.'
        return self._printer_type_name

    @pyqtProperty('QVariantList', notify=printJobsChanged)
    def printJobs(self) -> List[UM3PrintJobOutputModel]:
        if False:
            i = 10
            return i + 15
        return self._print_jobs

    @pyqtProperty('QVariantList', notify=printJobsChanged)
    def queuedPrintJobs(self) -> List[UM3PrintJobOutputModel]:
        if False:
            while True:
                i = 10
        return [print_job for print_job in self._print_jobs if print_job.state in self.QUEUED_PRINT_JOBS_STATES]

    @pyqtProperty('QVariantList', notify=printJobsChanged)
    def activePrintJobs(self) -> List[UM3PrintJobOutputModel]:
        if False:
            print('Hello World!')
        return [print_job for print_job in self._print_jobs if print_job.assignedPrinter is not None and print_job.state not in self.QUEUED_PRINT_JOBS_STATES]

    @pyqtProperty(bool, notify=_clusterPrintersChanged)
    def receivedData(self) -> bool:
        if False:
            print('Hello World!')
        return self._has_received_printers

    @pyqtProperty(int, notify=_clusterPrintersChanged)
    def clusterSize(self) -> int:
        if False:
            while True:
                i = 10
        if not self._has_received_printers:
            discovered_size = self.getProperty('cluster_size')
            if discovered_size == '':
                return 1
            return int(discovered_size)
        return len(self._printers)

    @pyqtProperty('QVariantList', notify=_clusterPrintersChanged)
    def connectedPrintersTypeCount(self) -> List[Dict[str, str]]:
        if False:
            i = 10
            return i + 15
        printer_count = {}
        for printer in self._printers:
            if printer.type in printer_count:
                printer_count[printer.type] += 1
            else:
                printer_count[printer.type] = 1
        result = []
        for machine_type in printer_count:
            result.append({'machine_type': machine_type, 'count': str(printer_count[machine_type])})
        return result

    @pyqtProperty('QVariantList', notify=_clusterPrintersChanged)
    def printers(self) -> List[PrinterOutputModel]:
        if False:
            while True:
                i = 10
        return self._printers

    @pyqtProperty(QObject, notify=activePrinterChanged)
    def activePrinter(self) -> Optional[PrinterOutputModel]:
        if False:
            for i in range(10):
                print('nop')
        return self._active_printer

    @pyqtSlot(QObject, name='setActivePrinter')
    def setActivePrinter(self, printer: Optional[PrinterOutputModel]) -> None:
        if False:
            while True:
                i = 10
        if self.activePrinter == printer:
            return
        self._active_printer = printer
        self.activePrinterChanged.emit()

    @pyqtProperty(bool, constant=True)
    def supportsPrintJobActions(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the printer that this output device represents supports print job actions via the local network.'
        return True

    def setJobState(self, print_job_uuid: str, state: str) -> None:
        if False:
            return 10
        'Set the remote print job state.'
        raise NotImplementedError('setJobState must be implemented')

    @pyqtSlot(str, name='sendJobToTop')
    def sendJobToTop(self, print_job_uuid: str) -> None:
        if False:
            return 10
        raise NotImplementedError('sendJobToTop must be implemented')

    @pyqtSlot(str, name='deleteJobFromQueue')
    def deleteJobFromQueue(self, print_job_uuid: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('deleteJobFromQueue must be implemented')

    @pyqtSlot(str, name='forceSendJob')
    def forceSendJob(self, print_job_uuid: str) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('forceSendJob must be implemented')

    @pyqtProperty(bool, constant=True)
    def supportsPrintJobQueue(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Whether this printer knows about queueing print jobs.\n        '
        return True

    @pyqtProperty(bool, constant=True)
    def canReadPrintJobs(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Whether this user can read the list of print jobs and their properties.\n        '
        return True

    @pyqtProperty(bool, constant=True)
    def canWriteOthersPrintJobs(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether this user can change things about print jobs made by other\n        people.\n        '
        return True

    @pyqtProperty(bool, constant=True)
    def canWriteOwnPrintJobs(self) -> bool:
        if False:
            while True:
                i = 10
        '\n        Whether this user can change things about print jobs made by themself.\n        '
        return True

    @pyqtProperty(bool, constant=True)
    def canReadPrinterDetails(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Whether this user can read the status of the printer.\n        '
        return True

    @pyqtSlot(name='openPrintJobControlPanel')
    def openPrintJobControlPanel(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('openPrintJobControlPanel must be implemented')

    @pyqtSlot(name='openPrinterControlPanel')
    def openPrinterControlPanel(self) -> None:
        if False:
            return 10
        raise NotImplementedError('openPrinterControlPanel must be implemented')

    @pyqtProperty(QUrl, notify=_clusterPrintersChanged)
    def activeCameraUrl(self) -> QUrl:
        if False:
            while True:
                i = 10
        return QUrl()

    @pyqtSlot(QUrl, name='setActiveCameraUrl')
    def setActiveCameraUrl(self, camera_url: QUrl) -> None:
        if False:
            return 10
        pass

    @pyqtSlot(int, result=str, name='getTimeCompleted')
    def getTimeCompleted(self, time_remaining: int) -> str:
        if False:
            i = 10
            return i + 15
        return formatTimeCompleted(time_remaining)

    @pyqtSlot(int, result=str, name='getDateCompleted')
    def getDateCompleted(self, time_remaining: int) -> str:
        if False:
            while True:
                i = 10
        return formatDateCompleted(time_remaining)

    @pyqtSlot(int, result=str, name='formatDuration')
    def formatDuration(self, seconds: int) -> str:
        if False:
            print('Hello World!')
        return Duration(seconds).getDisplayString(DurationFormat.Format.Short)

    def _update(self) -> None:
        if False:
            return 10
        super()._update()
        self._checkStillConnected()

    def _checkStillConnected(self) -> None:
        if False:
            while True:
                i = 10
        "Check if we're still connected by comparing the last timestamps for network response and the current time.\n\n        This implementation is similar to the base NetworkedPrinterOutputDevice, but is tweaked slightly.\n        Re-connecting is handled automatically by the output device managers in this plugin.\n        TODO: it would be nice to have this logic in the managers, but connecting those with signals causes crashes.\n        "
        time_since_last_response = time() - self._time_of_last_response
        if time_since_last_response > self._timeout_time:
            Logger.log('d', 'It has been %s seconds since the last response for outputdevice %s, so assume a timeout', time_since_last_response, self.key)
            self.setConnectionState(ConnectionState.Closed)
            if self.key in CuraApplication.getInstance().getOutputDeviceManager().getOutputDeviceIds():
                CuraApplication.getInstance().getOutputDeviceManager().removeOutputDevice(self.key)
        elif self.connectionState == ConnectionState.Closed:
            self._reconnectForActiveMachine()

    def _reconnectForActiveMachine(self) -> None:
        if False:
            while True:
                i = 10
        'Reconnect for the active output device.\n\n        Does nothing if the device is not meant for the active machine.\n        '
        active_machine = CuraApplication.getInstance().getGlobalContainerStack()
        if not active_machine:
            return
        Logger.log('d', 'Reconnecting output device after timeout.')
        self.setConnectionState(ConnectionState.Connected)
        if self.key in CuraApplication.getInstance().getOutputDeviceManager().getOutputDeviceIds():
            return
        stored_device_id = active_machine.getMetaDataEntry(self.META_NETWORK_KEY)
        if self.key == stored_device_id:
            CuraApplication.getInstance().getOutputDeviceManager().addOutputDevice(self)
        stored_cluster_id = active_machine.getMetaDataEntry(self.META_CLUSTER_ID)
        if self.key == stored_cluster_id:
            CuraApplication.getInstance().getOutputDeviceManager().addOutputDevice(self)

    def _responseReceived(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._time_of_last_response = time()

    def _updatePrinters(self, remote_printers: List[ClusterPrinterStatus]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._responseReceived()
        new_printers = []
        for (index, printer_data) in enumerate(remote_printers):
            printer = next(iter((printer for printer in self._printers if printer.key == printer_data.uuid)), None)
            if printer is None:
                printer = printer_data.createOutputModel(ClusterOutputController(self))
            else:
                printer_data.updateOutputModel(printer)
            new_printers.append(printer)
        remote_printers_keys = [printer_data.uuid for printer_data in remote_printers]
        removed_printers = [printer for printer in self._printers if printer.key not in remote_printers_keys]
        for removed_printer in removed_printers:
            if self._active_printer and self._active_printer.key == removed_printer.key:
                self.setActivePrinter(None)
        self._printers = new_printers
        self._has_received_printers = True
        if self._printers and (not self.activePrinter):
            self.setActivePrinter(self._printers[0])
        self.printersChanged.emit()
        self._checkIfClusterHost()

    def _checkIfClusterHost(self):
        if False:
            while True:
                i = 10
        'Check is this device is a cluster host and takes the needed actions when it is not.'
        if len(self._printers) < 1 and self.isConnected():
            self._num_is_host_check_failed += 1
        else:
            self._num_is_host_check_failed = 0
        if self._num_is_host_check_failed >= 6:
            NotClusterHostMessage(self).show()
            self.close()
            CuraApplication.getInstance().getOutputDeviceManager().removeOutputDevice(self.key)

    def _updatePrintJobs(self, remote_jobs: List[ClusterPrintJobStatus]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Updates the local list of print jobs with the list received from the cluster.\n\n        :param remote_jobs: The print jobs received from the cluster.\n        '
        self._responseReceived()
        new_print_jobs = []
        for print_job_data in remote_jobs:
            print_job = next(iter((print_job for print_job in self._print_jobs if print_job.key == print_job_data.uuid)), None)
            if not print_job:
                new_print_jobs.append(self._createPrintJobModel(print_job_data))
            else:
                print_job_data.updateOutputModel(print_job)
                if print_job_data.printer_uuid:
                    self._updateAssignedPrinter(print_job, print_job_data.printer_uuid)
                if print_job_data.assigned_to:
                    self._updateAssignedPrinter(print_job, print_job_data.assigned_to)
                new_print_jobs.append(print_job)
        remote_job_keys = [print_job_data.uuid for print_job_data in remote_jobs]
        removed_jobs = [print_job for print_job in self._print_jobs if print_job.key not in remote_job_keys]
        for removed_job in removed_jobs:
            if removed_job.assignedPrinter:
                removed_job.assignedPrinter.updateActivePrintJob(None)
        self._print_jobs = new_print_jobs
        self.printJobsChanged.emit()

    def _createPrintJobModel(self, remote_job: ClusterPrintJobStatus) -> UM3PrintJobOutputModel:
        if False:
            while True:
                i = 10
        'Create a new print job model based on the remote status of the job.\n\n        :param remote_job: The remote print job data.\n        '
        model = remote_job.createOutputModel(ClusterOutputController(self))
        if remote_job.printer_uuid:
            self._updateAssignedPrinter(model, remote_job.printer_uuid)
        if remote_job.assigned_to:
            self._updateAssignedPrinter(model, remote_job.assigned_to)
        if remote_job.preview_url:
            model.loadPreviewImageFromUrl(remote_job.preview_url)
        return model

    def _updateAssignedPrinter(self, model: UM3PrintJobOutputModel, printer_uuid: str) -> None:
        if False:
            while True:
                i = 10
        'Updates the printer assignment for the given print job model.'
        printer = next((p for p in self._printers if printer_uuid == p.key), None)
        if not printer:
            return
        printer.updateActivePrintJob(model)
        model.updateAssignedPrinter(printer)

    def _loadMonitorTab(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Load Monitor tab QML.'
        plugin_registry = CuraApplication.getInstance().getPluginRegistry()
        if not plugin_registry:
            Logger.log('e', 'Could not get plugin registry')
            return
        plugin_path = plugin_registry.getPluginPath('UM3NetworkPrinting')
        if not plugin_path:
            Logger.log('e', 'Could not get plugin path')
            return
        self._monitor_view_qml_path = os.path.join(plugin_path, 'resources', 'qml', 'MonitorStage.qml')