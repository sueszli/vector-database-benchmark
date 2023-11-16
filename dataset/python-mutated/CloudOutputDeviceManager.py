import os
from typing import Dict, List, Optional, Set
from PyQt6.QtNetwork import QNetworkReply
from PyQt6.QtWidgets import QMessageBox
from UM import i18nCatalog
from UM.Logger import Logger
from UM.Message import Message
from UM.Settings.Interfaces import ContainerInterface
from UM.Signal import Signal
from UM.Util import parseBool
from cura.API import Account
from cura.API.Account import SyncState
from cura.CuraApplication import CuraApplication
from cura.Settings.CuraContainerRegistry import CuraContainerRegistry
from cura.Settings.CuraStackBuilder import CuraStackBuilder
from cura.Settings.GlobalStack import GlobalStack
from cura.UltimakerCloud.UltimakerCloudConstants import META_CAPABILITIES, META_UM_LINKED_TO_ACCOUNT
from .AbstractCloudOutputDevice import AbstractCloudOutputDevice
from .CloudApiClient import CloudApiClient
from .CloudOutputDevice import CloudOutputDevice
from ..Messages.RemovedPrintersMessage import RemovedPrintersMessage
from ..Models.Http.CloudClusterResponse import CloudClusterResponse
from ..Messages.NewPrinterDetectedMessage import NewPrinterDetectedMessage
catalog = i18nCatalog('cura')

class CloudOutputDeviceManager:
    """The cloud output device manager is responsible for using the Ultimaker Cloud APIs to manage remote clusters.

    Keeping all cloud related logic in this class instead of the UM3OutputDevicePlugin results in more readable code.
    API spec is available on https://docs.api.ultimaker.com/connect/index.html.
    """
    META_CLUSTER_ID = 'um_cloud_cluster_id'
    META_HOST_GUID = 'host_guid'
    META_NETWORK_KEY = 'um_network_key'
    SYNC_SERVICE_NAME = 'CloudOutputDeviceManager'
    i18n_catalog = i18nCatalog('cura')
    discoveredDevicesChanged = Signal()

    def __init__(self) -> None:
        if False:
            return 10
        self._remote_clusters: Dict[str, CloudOutputDevice] = {}
        self._abstract_clusters: Dict[str, AbstractCloudOutputDevice] = {}
        self._um_cloud_printers: Dict[str, GlobalStack] = {}
        self._account: Account = CuraApplication.getInstance().getCuraAPI().account
        self._api = CloudApiClient(CuraApplication.getInstance(), on_error=lambda error: Logger.log('e', str(error)))
        self._account.loginStateChanged.connect(self._onLoginStateChanged)
        self._removed_printers_message: Optional[RemovedPrintersMessage] = None
        self._running = False
        self._syncing = False
        CuraApplication.getInstance().getContainerRegistry().containerRemoved.connect(self._printerRemoved)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Starts running the cloud output device manager, thus periodically requesting cloud data.'
        if self._running:
            return
        if not self._account.isLoggedIn:
            return
        self._running = True
        self._getRemoteClusters()
        self._account.syncRequested.connect(self._getRemoteClusters)

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        'Stops running the cloud output device manager.'
        if not self._running:
            return
        self._running = False
        self._onGetRemoteClustersFinished([])

    def refreshConnections(self) -> None:
        if False:
            return 10
        'Force refreshing connections.'
        self._connectToActiveMachine()

    def _onLoginStateChanged(self, is_logged_in: bool) -> None:
        if False:
            print('Hello World!')
        'Called when the uses logs in or out'
        if is_logged_in:
            self.start()
        else:
            self.stop()

    def _getRemoteClusters(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Gets all remote clusters from the API.'
        if self._syncing:
            return
        self._syncing = True
        self._account.setSyncState(self.SYNC_SERVICE_NAME, SyncState.SYNCING)
        self._api.getClusters(self._onGetRemoteClustersFinished, self._onGetRemoteClusterFailed)

    def _onGetRemoteClustersFinished(self, clusters: List[CloudClusterResponse]) -> None:
        if False:
            while True:
                i = 10
        'Callback for when the request for getting the clusters is successful and finished.'
        self._um_cloud_printers = {m.getMetaDataEntry(self.META_CLUSTER_ID): m for m in CuraApplication.getInstance().getContainerRegistry().findContainerStacks(type='machine') if m.getMetaDataEntry(self.META_CLUSTER_ID, None)}
        new_clusters = []
        all_clusters: Dict[str, CloudClusterResponse] = {c.cluster_id: c for c in clusters}
        online_clusters: Dict[str, CloudClusterResponse] = {c.cluster_id: c for c in clusters if c.is_online}
        for (device_id, cluster_data) in all_clusters.items():
            if device_id not in self._remote_clusters:
                new_clusters.append(cluster_data)
            if device_id in self._um_cloud_printers:
                if not self._um_cloud_printers[device_id].getMetaDataEntry(self.META_HOST_GUID, None):
                    self._um_cloud_printers[device_id].setMetaDataEntry(self.META_HOST_GUID, cluster_data.host_guid)
                if not parseBool(self._um_cloud_printers[device_id].getMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, 'true')):
                    self._um_cloud_printers[device_id].setMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, True)
                if not self._um_cloud_printers[device_id].getMetaDataEntry(META_CAPABILITIES, None):
                    self._um_cloud_printers[device_id].setMetaDataEntry(META_CAPABILITIES, ','.join(cluster_data.capabilities))
        self._createMachineStacksForDiscoveredClusters(new_clusters)
        self._updateOnlinePrinters(all_clusters)
        if self._removed_printers_message:
            self._removed_printers_message.actionTriggered.disconnect(self._onRemovedPrintersMessageActionTriggered)
            self._removed_printers_message.hide()
        offline_device_keys = set(self._remote_clusters.keys()) - set(online_clusters.keys())
        for device_id in offline_device_keys:
            self._onDiscoveredDeviceRemoved(device_id)
        removed_device_keys = set(self._um_cloud_printers.keys()) - set(all_clusters.keys())
        if removed_device_keys:
            self._devicesRemovedFromAccount(removed_device_keys)
        if new_clusters or offline_device_keys or removed_device_keys:
            self.discoveredDevicesChanged.emit()
        if offline_device_keys:
            self._connectToActiveMachine()
        self._syncing = False
        self._account.setSyncState(self.SYNC_SERVICE_NAME, SyncState.SUCCESS)
        Logger.debug('Synced cloud printers with account.')

    def _onGetRemoteClusterFailed(self, reply: QNetworkReply, error: QNetworkReply.NetworkError) -> None:
        if False:
            while True:
                i = 10
        self._syncing = False
        self._account.setSyncState(self.SYNC_SERVICE_NAME, SyncState.ERROR)

    def _requestWrite(self, unique_id: str, nodes: List['SceneNode']):
        if False:
            print('Hello World!')
        for remote in self._remote_clusters.values():
            if unique_id == remote.name:
                remote.requestWrite(nodes)
                return
        Logger.log('e', f'Failed writing to specific cloud printer: {unique_id} not in remote clusters.')
        message = Message(catalog.i18nc('@info:status', 'Failed writing to specific cloud printer: {0} not in remote clusters.').format(unique_id), title=catalog.i18nc('@info:title', 'Error'), message_type=Message.MessageType.ERROR)
        message.show()

    def _createMachineStacksForDiscoveredClusters(self, discovered_clusters: List[CloudClusterResponse]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '**Synchronously** create machines for discovered devices\n\n        Any new machines are made available to the user.\n        May take a long time to complete. This currently forcefully calls the "processEvents", which isn\'t\n        the nicest solution out there. We might need to consider moving this into a job later!\n        '
        new_output_devices: List[CloudOutputDevice] = []
        remote_clusters_added = False
        host_guid_map: Dict[str, str] = {machine.getMetaDataEntry(self.META_HOST_GUID): device_cluster_id for (device_cluster_id, machine) in self._um_cloud_printers.items() if machine.getMetaDataEntry(self.META_HOST_GUID)}
        machine_manager = CuraApplication.getInstance().getMachineManager()
        for cluster_data in discovered_clusters:
            output_device = CloudOutputDevice(self._api, cluster_data)
            if cluster_data.printer_type not in self._abstract_clusters:
                self._abstract_clusters[cluster_data.printer_type] = AbstractCloudOutputDevice(self._api, cluster_data.printer_type, self._requestWrite, self.refreshConnections)
                _abstract_machine = CuraStackBuilder.createAbstractMachine(cluster_data.printer_type)
            if cluster_data.host_guid in host_guid_map:
                machine = machine_manager.getMachine(output_device.printerType, {self.META_HOST_GUID: cluster_data.host_guid})
                if machine and machine.getMetaDataEntry(self.META_CLUSTER_ID) != output_device.key:
                    self._updateOutdatedMachine(outdated_machine=machine, new_cloud_output_device=output_device)
            if machine_manager.getMachine(output_device.printerType, {self.META_CLUSTER_ID: output_device.key}) is None and machine_manager.getMachine(output_device.printerType, {self.META_NETWORK_KEY: cluster_data.host_name + '*'}) is None:
                new_output_devices.append(output_device)
            elif output_device.getId() not in self._remote_clusters:
                self._remote_clusters[output_device.getId()] = output_device
                remote_clusters_added = True
            elif not parseBool(self._um_cloud_printers[output_device.key].getMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, 'true')):
                self._um_cloud_printers[output_device.key].setMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, True)
            CuraApplication.getInstance().processEvents()
        new_devices_list_of_dicts = [{'key': d.getId(), 'name': d.name, 'machine_type': d.printerTypeName, 'firmware_version': d.firmwareVersion} for d in new_output_devices]
        discovered_cloud_printers_model = CuraApplication.getInstance().getDiscoveredCloudPrintersModel()
        discovered_cloud_printers_model.addDiscoveredCloudPrinters(new_devices_list_of_dicts)
        if not new_output_devices:
            if remote_clusters_added:
                self._connectToActiveMachine()
            return
        online_cluster_names = {c.friendly_name.lower() for c in discovered_clusters if c.is_online and (not c.friendly_name is None)}
        new_output_devices.sort(key=lambda x: ('a{}' if x.name.lower() in online_cluster_names else 'b{}').format(x.name.lower()))
        message = NewPrinterDetectedMessage(num_printers_found=len(new_output_devices))
        message.show()
        new_devices_added = []
        for (idx, output_device) in enumerate(new_output_devices):
            message.updateProgressText(output_device)
            self._remote_clusters[output_device.getId()] = output_device
            activate = not CuraApplication.getInstance().getMachineManager().activeMachine
            if self._createMachineFromDiscoveredDevice(output_device.getId(), activate=activate):
                new_devices_added.append(output_device)
        message.finalize(new_devices_added, new_output_devices)

    @staticmethod
    def _updateOnlinePrinters(printer_responses: Dict[str, CloudClusterResponse]) -> None:
        if False:
            while True:
                i = 10
        '\n        Update the metadata of the printers to store whether they are online or not.\n        :param printer_responses: The responses received from the API about the printer statuses.\n        '
        for container_stack in CuraContainerRegistry.getInstance().findContainerStacks(type='machine'):
            cluster_id = container_stack.getMetaDataEntry('um_cloud_cluster_id', '')
            if cluster_id in printer_responses:
                container_stack.setMetaDataEntry('is_online', printer_responses[cluster_id].is_online)

    def _updateOutdatedMachine(self, outdated_machine: GlobalStack, new_cloud_output_device: CloudOutputDevice) -> None:
        if False:
            return 10
        '\n         Update the cloud metadata of a pre-existing machine that is rediscovered (e.g. if the printer was removed and\n         re-added to the account) and delete the old CloudOutputDevice related to this machine.\n\n        :param outdated_machine: The cloud machine that needs to be brought up-to-date with the new data received from\n                                 the account\n        :param new_cloud_output_device: The new CloudOutputDevice that should be linked to the pre-existing machine\n        :return: None\n        '
        old_cluster_id = outdated_machine.getMetaDataEntry(self.META_CLUSTER_ID)
        outdated_machine.setMetaDataEntry(self.META_CLUSTER_ID, new_cloud_output_device.key)
        outdated_machine.setMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, True)
        self._um_cloud_printers[new_cloud_output_device.key] = self._um_cloud_printers.pop(old_cluster_id)
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        if old_cluster_id in output_device_manager.getOutputDeviceIds():
            output_device_manager.removeOutputDevice(old_cluster_id)
        if old_cluster_id in self._remote_clusters:
            self._remote_clusters[old_cluster_id].close()
            del self._remote_clusters[old_cluster_id]
            self._remote_clusters[new_cloud_output_device.key] = new_cloud_output_device

    def _devicesRemovedFromAccount(self, removed_device_ids: Set[str]) -> None:
        if False:
            return 10
        '\n        Removes the CloudOutputDevice from the received device ids and marks the specific printers as "removed from\n        account". In addition, it generates a message to inform the user about the printers that are no longer linked to\n        his/her account. The message is not generated if all the printers have been previously reported as not linked\n        to the account.\n\n        :param removed_device_ids: Set of device ids, whose CloudOutputDevice needs to be removed\n        :return: None\n        '
        if not CuraApplication.getInstance().getCuraAPI().account.isLoggedIn:
            return
        ignored_device_ids = set()
        for device_id in removed_device_ids:
            if not parseBool(self._um_cloud_printers[device_id].getMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, 'true')):
                ignored_device_ids.add(device_id)
        self.reported_device_ids = removed_device_ids - ignored_device_ids
        if not self.reported_device_ids:
            return
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        for device_id in removed_device_ids:
            global_stack: Optional[GlobalStack] = self._um_cloud_printers.get(device_id, None)
            if not global_stack:
                continue
            if device_id in output_device_manager.getOutputDeviceIds():
                output_device_manager.removeOutputDevice(device_id)
            if device_id in self._remote_clusters:
                del self._remote_clusters[device_id]
            global_stack.setMetaDataEntry(META_UM_LINKED_TO_ACCOUNT, False)
        device_names = ''.join(['<li>{} ({})</li>'.format(self._um_cloud_printers[device].name, self._um_cloud_printers[device].definition.name) for device in self.reported_device_ids])
        self._removed_printers_message = RemovedPrintersMessage(self.reported_device_ids, device_names)
        self._removed_printers_message.actionTriggered.connect(self._onRemovedPrintersMessageActionTriggered)
        self._removed_printers_message.show()

    def _onDiscoveredDeviceRemoved(self, device_id: str) -> None:
        if False:
            print('Hello World!')
        ' Remove the CloudOutputDevices for printers that are offline'
        device: Optional[CloudOutputDevice] = self._remote_clusters.pop(device_id, None)
        if not device:
            return
        device.close()
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        if device.key in output_device_manager.getOutputDeviceIds():
            output_device_manager.removeOutputDevice(device.key)

    def _createMachineFromDiscoveredDevice(self, key: str, activate: bool=True) -> bool:
        if False:
            print('Hello World!')
        device = self._remote_clusters.get(key)
        if not device:
            return False
        new_machine = CuraStackBuilder.createMachine(device.name, device.printerType, show_warning_message=False)
        if not new_machine:
            Logger.error(f'Failed creating a new machine for {device.name}')
            return False
        self._setOutputDeviceMetadata(device, new_machine)
        if activate:
            CuraApplication.getInstance().getMachineManager().setActiveMachine(new_machine.getId())
        return True

    def _connectToActiveMachine(self) -> None:
        if False:
            return 10
        'Callback for when the active machine was changed by the user'
        active_machine = CuraApplication.getInstance().getGlobalContainerStack()
        if not active_machine:
            return
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        stored_cluster_id = active_machine.getMetaDataEntry(self.META_CLUSTER_ID)
        local_network_key = active_machine.getMetaDataEntry(self.META_NETWORK_KEY)
        remote_cluster_copy: List[CloudOutputDevice] = list(self._remote_clusters.values())
        for device in remote_cluster_copy:
            if device.key == stored_cluster_id:
                self._connectToOutputDevice(device, active_machine)
            elif local_network_key and device.matchesNetworkKey(local_network_key):
                self._connectToOutputDevice(device, active_machine)
            elif device.key in output_device_manager.getOutputDeviceIds():
                output_device_manager.removeOutputDevice(device.key)
        remote_abstract_cluster_copy: List[CloudOutputDevice] = list(self._abstract_clusters.values())
        for device in remote_abstract_cluster_copy:
            if device.printerType == active_machine.definition.getId() and parseBool(active_machine.getMetaDataEntry('is_abstract_machine', False)):
                self._connectToAbstractOutputDevice(device, active_machine)
            elif device.key in output_device_manager.getOutputDeviceIds():
                output_device_manager.removeOutputDevice(device.key)

    def _setOutputDeviceMetadata(self, device: CloudOutputDevice, machine: GlobalStack):
        if False:
            while True:
                i = 10
        machine.setName(device.name)
        machine.setMetaDataEntry(self.META_CLUSTER_ID, device.key)
        machine.setMetaDataEntry(self.META_HOST_GUID, device.clusterData.host_guid)
        machine.setMetaDataEntry('group_name', device.name)
        machine.setMetaDataEntry('group_size', device.clusterSize)
        digital_factory_string = self.i18n_catalog.i18nc('info:name', 'Ultimaker Digital Factory')
        digital_factory_link = f"<a href='https://digitalfactory.ultimaker.com?utm_source=cura&utm_medium=software&utm_campaign=change-account-remove-printer'>{digital_factory_string}</a>"
        removal_warning_string = self.i18n_catalog.i18nc('@message {printer_name} is replaced with the name of the printer', '{printer_name} will be removed until the next account sync.').format(printer_name=device.name) + '<br>' + self.i18n_catalog.i18nc('@message {printer_name} is replaced with the name of the printer', 'To remove {printer_name} permanently, visit {digital_factory_link}').format(printer_name=device.name, digital_factory_link=digital_factory_link) + '<br><br>' + self.i18n_catalog.i18nc('@message {printer_name} is replaced with the name of the printer', 'Are you sure you want to remove {printer_name} temporarily?').format(printer_name=device.name)
        machine.setMetaDataEntry('removal_warning', removal_warning_string)
        machine.addConfiguredConnectionType(device.connectionType.value)

    def _connectToAbstractOutputDevice(self, device: AbstractCloudOutputDevice, machine: GlobalStack) -> None:
        if False:
            i = 10
            return i + 15
        Logger.debug(f'Attempting to connect to abstract machine {machine.id}')
        if not device.isConnected():
            device.connect()
        machine.addConfiguredConnectionType(device.connectionType.value)
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        if device.key not in output_device_manager.getOutputDeviceIds():
            output_device_manager.addOutputDevice(device)

    def _connectToOutputDevice(self, device: CloudOutputDevice, machine: GlobalStack) -> None:
        if False:
            return 10
        'Connects to an output device and makes sure it is registered in the output device manager.'
        self._setOutputDeviceMetadata(device, machine)
        if not device.isConnected():
            device.connect()
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        if device.key not in output_device_manager.getOutputDeviceIds():
            output_device_manager.addOutputDevice(device)

    def _printerRemoved(self, container: ContainerInterface) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Callback connected to the containerRemoved signal. Invoked when a cloud printer is removed from Cura to remove\n        the printer's reference from the _remote_clusters.\n\n        :param container: The ContainerInterface passed to this function whenever the ContainerRemoved signal is emitted\n        :return: None\n        "
        if isinstance(container, GlobalStack):
            container_cluster_id = container.getMetaDataEntry(self.META_CLUSTER_ID, None)
            if container_cluster_id in self._remote_clusters.keys():
                del self._remote_clusters[container_cluster_id]

    def _onRemovedPrintersMessageActionTriggered(self, removed_printers_message: RemovedPrintersMessage, action: str) -> None:
        if False:
            while True:
                i = 10
        if action == 'keep_printer_configurations_action':
            removed_printers_message.hide()
        elif action == 'remove_printers_action':
            machine_manager = CuraApplication.getInstance().getMachineManager()
            remove_printers_ids = {self._um_cloud_printers[i].getId() for i in self.reported_device_ids}
            all_ids = {m.getId() for m in CuraApplication.getInstance().getContainerRegistry().findContainerStacks(type='machine')}
            question_title = self.i18n_catalog.i18nc('@title:window', 'Remove printers?')
            question_content = self.i18n_catalog.i18ncp('@label', 'You are about to remove {0} printer from Cura. This action cannot be undone.\nAre you sure you want to continue?', 'You are about to remove {0} printers from Cura. This action cannot be undone.\nAre you sure you want to continue?', len(remove_printers_ids))
            if remove_printers_ids == all_ids:
                question_content = self.i18n_catalog.i18nc('@label', 'You are about to remove all printers from Cura. This action cannot be undone.\nAre you sure you want to continue?')
            result = QMessageBox.question(None, question_title, question_content)
            if result == QMessageBox.StandardButton.No:
                return
            for machine_cloud_id in self.reported_device_ids:
                machine_manager.setActiveMachine(self._um_cloud_printers[machine_cloud_id].getId())
                machine_manager.removeMachine(self._um_cloud_printers[machine_cloud_id].getId())
            removed_printers_message.hide()