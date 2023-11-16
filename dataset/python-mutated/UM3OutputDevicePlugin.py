from typing import Optional, Callable, Dict
from UM.Signal import Signal
from cura.CuraApplication import CuraApplication
from UM.OutputDevice.OutputDeviceManager import ManualDeviceAdditionAttempt
from UM.OutputDevice.OutputDevicePlugin import OutputDevicePlugin
from .Network.LocalClusterOutputDevice import LocalClusterOutputDevice
from .Network.LocalClusterOutputDeviceManager import LocalClusterOutputDeviceManager
from .Cloud.CloudOutputDeviceManager import CloudOutputDeviceManager

class UM3OutputDevicePlugin(OutputDevicePlugin):
    """This plugin handles the discovery and networking for Ultimaker 3D printers"""
    discoveredDevicesChanged = Signal()
    'Signal emitted when the list of discovered devices changed. Used by printer action in this plugin.'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._network_output_device_manager = LocalClusterOutputDeviceManager()
        self._network_output_device_manager.discoveredDevicesChanged.connect(self.discoveredDevicesChanged)
        self._cloud_output_device_manager = CloudOutputDeviceManager()
        CuraApplication.getInstance().globalContainerStackChanged.connect(self.refreshConnections)

    def start(self):
        if False:
            while True:
                i = 10
        'Start looking for devices in the network and cloud.'
        self._network_output_device_manager.start()
        self._cloud_output_device_manager.start()

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        self._network_output_device_manager.stop()
        self._cloud_output_device_manager.stop()

    def startDiscovery(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Restart network discovery.'
        self._network_output_device_manager.startDiscovery()

    def refreshConnections(self) -> None:
        if False:
            return 10
        'Force refreshing the network connections.'
        self._network_output_device_manager.refreshConnections()
        self._cloud_output_device_manager.refreshConnections()

    def canAddManualDevice(self, address: str='') -> ManualDeviceAdditionAttempt:
        if False:
            i = 10
            return i + 15
        'Indicate that this plugin supports adding networked printers manually.'
        return ManualDeviceAdditionAttempt.PRIORITY

    def addManualDevice(self, address: str, callback: Optional[Callable[[bool, str], None]]=None) -> None:
        if False:
            return 10
        'Add a networked printer manually based on its network address.'
        self._network_output_device_manager.addManualDevice(address, callback)

    def removeManualDevice(self, key: str, address: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Remove a manually connected networked printer.'
        self._network_output_device_manager.removeManualDevice(key, address)

    def getDiscoveredDevices(self) -> Dict[str, LocalClusterOutputDevice]:
        if False:
            return 10
        'Get the discovered devices from the local network.'
        return self._network_output_device_manager.getDiscoveredDevices()

    def associateActiveMachineWithPrinterDevice(self, device: LocalClusterOutputDevice) -> None:
        if False:
            while True:
                i = 10
        'Connect the active machine to a device.'
        self._network_output_device_manager.associateActiveMachineWithPrinterDevice(device)