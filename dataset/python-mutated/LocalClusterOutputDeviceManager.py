from typing import Dict, Optional, Callable, List
from UM import i18nCatalog
from UM.Logger import Logger
from UM.Signal import Signal
from UM.Version import Version
from cura.CuraApplication import CuraApplication
from cura.Settings.CuraStackBuilder import CuraStackBuilder
from cura.Settings.GlobalStack import GlobalStack
from .ZeroConfClient import ZeroConfClient
from .ClusterApiClient import ClusterApiClient
from .LocalClusterOutputDevice import LocalClusterOutputDevice
from ..UltimakerNetworkedPrinterOutputDevice import UltimakerNetworkedPrinterOutputDevice
from ..Messages.CloudFlowMessage import CloudFlowMessage
from ..Messages.LegacyDeviceNoLongerSupportedMessage import LegacyDeviceNoLongerSupportedMessage
from ..Models.Http.PrinterSystemStatus import PrinterSystemStatus
I18N_CATALOG = i18nCatalog('cura')

class LocalClusterOutputDeviceManager:
    """The LocalClusterOutputDeviceManager is responsible for discovering and managing local networked clusters."""
    META_NETWORK_KEY = 'um_network_key'
    MANUAL_DEVICES_PREFERENCE_KEY = 'um3networkprinting/manual_instances'
    MIN_SUPPORTED_CLUSTER_VERSION = Version('4.0.0')
    I18N_CATALOG = i18nCatalog('cura')
    discoveredDevicesChanged = Signal()

    def __init__(self) -> None:
        if False:
            return 10
        self._discovered_devices = {}
        self._output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        self._zero_conf_client = ZeroConfClient()
        self._zero_conf_client.addedNetworkCluster.connect(self._onDeviceDiscovered)
        self._zero_conf_client.removedNetworkCluster.connect(self._onDiscoveredDeviceRemoved)

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        'Start the network discovery.'
        self._zero_conf_client.start()
        for address in self._getStoredManualAddresses():
            self.addManualDevice(address)

    def stop(self) -> None:
        if False:
            return 10
        'Stop network discovery and clean up discovered devices.'
        self._zero_conf_client.stop()
        for instance_name in list(self._discovered_devices):
            self._onDiscoveredDeviceRemoved(instance_name)

    def startDiscovery(self):
        if False:
            i = 10
            return i + 15
        'Restart discovery on the local network.'
        self.stop()
        self.start()

    def addManualDevice(self, address: str, callback: Optional[Callable[[bool, str], None]]=None) -> None:
        if False:
            while True:
                i = 10
        'Add a networked printer manually by address.'
        api_client = ClusterApiClient(address, lambda error: Logger.log('e', str(error)))
        api_client.getSystem(lambda status: self._onCheckManualDeviceResponse(address, status, callback))

    def removeManualDevice(self, device_id: str, address: Optional[str]=None) -> None:
        if False:
            return 10
        'Remove a manually added networked printer.'
        if device_id not in self._discovered_devices and address is not None:
            device_id = 'manual:{}'.format(address)
        if device_id in self._discovered_devices:
            address = address or self._discovered_devices[device_id].ipAddress
            self._onDiscoveredDeviceRemoved(device_id)
        if address in self._getStoredManualAddresses():
            self._removeStoredManualAddress(address)

    def refreshConnections(self) -> None:
        if False:
            return 10
        'Force reset all network device connections.'
        self._connectToActiveMachine()

    def getDiscoveredDevices(self) -> Dict[str, LocalClusterOutputDevice]:
        if False:
            i = 10
            return i + 15
        'Get the discovered devices.'
        return self._discovered_devices

    def associateActiveMachineWithPrinterDevice(self, device: LocalClusterOutputDevice) -> None:
        if False:
            return 10
        'Connect the active machine to a given device.'
        active_machine = CuraApplication.getInstance().getGlobalContainerStack()
        if not active_machine:
            return
        self._connectToOutputDevice(device, active_machine)
        self._connectToActiveMachine()
        definitions = CuraApplication.getInstance().getContainerRegistry().findContainers(id=device.printerType)
        if not definitions:
            return
        CuraApplication.getInstance().getMachineManager().switchPrinterType(definitions[0].getName())

    def _connectToActiveMachine(self) -> None:
        if False:
            return 10
        'Callback for when the active machine was changed by the user or a new remote cluster was found.'
        active_machine = CuraApplication.getInstance().getGlobalContainerStack()
        if not active_machine:
            return
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        stored_device_id = active_machine.getMetaDataEntry(self.META_NETWORK_KEY)
        for device in self._discovered_devices.values():
            if device.key == stored_device_id:
                self._connectToOutputDevice(device, active_machine)
            elif device.key in output_device_manager.getOutputDeviceIds():
                output_device_manager.removeOutputDevice(device.key)

    def _onCheckManualDeviceResponse(self, address: str, status: PrinterSystemStatus, callback: Optional[Callable[[bool, str], None]]=None) -> None:
        if False:
            print('Hello World!')
        'Callback for when a manual device check request was responded to.'
        self._onDeviceDiscovered('manual:{}'.format(address), address, {b'name': status.name.encode('utf-8'), b'address': address.encode('utf-8'), b'machine': str(status.hardware.get('typeid', '')).encode('utf-8'), b'manual': b'true', b'firmware_version': status.firmware.encode('utf-8'), b'cluster_size': b'1'})
        self._storeManualAddress(address)
        if callback is not None:
            CuraApplication.getInstance().callLater(callback, True, address)

    @staticmethod
    def _getPrinterTypeIdentifiers() -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        'Returns a dict of printer BOM numbers to machine types.\n\n        These numbers are available in the machine definition already so we just search for them here.\n        '
        container_registry = CuraApplication.getInstance().getContainerRegistry()
        ultimaker_machines = container_registry.findContainersMetadata(type='machine', manufacturer='Ultimaker B.V.')
        found_machine_type_identifiers = {}
        for machine in ultimaker_machines:
            machine_type = machine.get('id', None)
            machine_bom_numbers = machine.get('bom_numbers', [])
            if machine_type and machine_bom_numbers:
                for bom_number in machine_bom_numbers:
                    found_machine_type_identifiers[str(bom_number)] = machine_type
        return found_machine_type_identifiers

    def _onDeviceDiscovered(self, key: str, address: str, properties: Dict[bytes, bytes]) -> None:
        if False:
            print('Hello World!')
        'Add a new device.'
        machine_identifier = properties.get(b'machine', b'').decode('utf-8')
        printer_type_identifiers = self._getPrinterTypeIdentifiers()
        properties[b'printer_type'] = b'Unknown'
        for (bom, p_type) in printer_type_identifiers.items():
            if machine_identifier.startswith(bom):
                properties[b'printer_type'] = bytes(p_type, encoding='utf8')
                break
        device = LocalClusterOutputDevice(key, address, properties)
        discovered_printers_model = CuraApplication.getInstance().getDiscoveredPrintersModel()
        if address in list(discovered_printers_model.discoveredPrintersByAddress.keys()):
            discovered_printers_model.updateDiscoveredPrinter(ip_address=address, name=device.getName(), machine_type=device.printerType)
        else:
            discovered_printers_model.addDiscoveredPrinter(ip_address=address, key=device.getId(), name=device.getName(), create_callback=self._createMachineFromDiscoveredDevice, machine_type=device.printerType, device=device)
        self._discovered_devices[device.getId()] = device
        self.discoveredDevicesChanged.emit()
        self._connectToActiveMachine()

    def _onDiscoveredDeviceRemoved(self, device_id: str) -> None:
        if False:
            print('Hello World!')
        'Remove a device.'
        device = self._discovered_devices.pop(device_id, None)
        if not device:
            return
        device.close()
        CuraApplication.getInstance().getDiscoveredPrintersModel().removeDiscoveredPrinter(device.address)
        self.discoveredDevicesChanged.emit()

    def _createMachineFromDiscoveredDevice(self, device_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create a machine instance based on the discovered network printer.'
        device = self._discovered_devices.get(device_id)
        if device is None:
            return
        new_machine = CuraStackBuilder.createMachine(device.name, device.printerType)
        if not new_machine:
            Logger.log('e', 'Failed creating a new machine')
            return
        new_machine.setMetaDataEntry(self.META_NETWORK_KEY, device.key)
        CuraApplication.getInstance().getMachineManager().setActiveMachine(new_machine.getId())
        self._connectToOutputDevice(device, new_machine)
        self._showCloudFlowMessage(device)
        _abstract_machine = CuraStackBuilder.createAbstractMachine(device.printerType)

    def _storeManualAddress(self, address: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add an address to the stored preferences.'
        stored_addresses = self._getStoredManualAddresses()
        if address in stored_addresses:
            return
        stored_addresses.append(address)
        new_value = ','.join(stored_addresses)
        CuraApplication.getInstance().getPreferences().setValue(self.MANUAL_DEVICES_PREFERENCE_KEY, new_value)

    def _removeStoredManualAddress(self, address: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Remove an address from the stored preferences.'
        stored_addresses = self._getStoredManualAddresses()
        try:
            stored_addresses.remove(address)
            new_value = ','.join(stored_addresses)
            CuraApplication.getInstance().getPreferences().setValue(self.MANUAL_DEVICES_PREFERENCE_KEY, new_value)
        except ValueError:
            Logger.log('w', 'Could not remove address from stored_addresses, it was not there')

    def _getStoredManualAddresses(self) -> List[str]:
        if False:
            while True:
                i = 10
        'Load the user-configured manual devices from Cura preferences.'
        preferences = CuraApplication.getInstance().getPreferences()
        preferences.addPreference(self.MANUAL_DEVICES_PREFERENCE_KEY, '')
        manual_instances = preferences.getValue(self.MANUAL_DEVICES_PREFERENCE_KEY).split(',')
        return manual_instances

    def _connectToOutputDevice(self, device: UltimakerNetworkedPrinterOutputDevice, machine: GlobalStack) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Add a device to the current active machine.'
        if Version(device.firmwareVersion) < self.MIN_SUPPORTED_CLUSTER_VERSION:
            LegacyDeviceNoLongerSupportedMessage().show()
            return
        machine.setName(device.name)
        machine.setMetaDataEntry(self.META_NETWORK_KEY, device.key)
        machine.setMetaDataEntry('group_name', device.name)
        machine.addConfiguredConnectionType(device.connectionType.value)
        if not device.isConnected():
            device.connect()
        output_device_manager = CuraApplication.getInstance().getOutputDeviceManager()
        if device.key not in output_device_manager.getOutputDeviceIds():
            output_device_manager.addOutputDevice(device)

    @staticmethod
    def _showCloudFlowMessage(device: LocalClusterOutputDevice) -> None:
        if False:
            return 10
        'Nudge the user to start using Ultimaker Cloud.'
        if CuraApplication.getInstance().getMachineManager().activeMachineHasCloudRegistration:
            return
        if not CuraApplication.getInstance().getCuraAPI().account.isLoggedIn:
            return
        CloudFlowMessage(device.name).show()