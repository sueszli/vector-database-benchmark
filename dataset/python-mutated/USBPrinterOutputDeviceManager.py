import threading
import time
import serial.tools.list_ports
from os import environ
from re import search
from PyQt6.QtCore import QObject, pyqtSignal
from UM.Platform import Platform
from UM.Signal import Signal, signalemitter
from UM.OutputDevice.OutputDevicePlugin import OutputDevicePlugin
from UM.i18n import i18nCatalog
from cura.PrinterOutput.PrinterOutputDevice import ConnectionState
from . import USBPrinterOutputDevice
i18n_catalog = i18nCatalog('cura')

@signalemitter
class USBPrinterOutputDeviceManager(QObject, OutputDevicePlugin):
    """Manager class that ensures that an USBPrinterOutput device is created for every connected USB printer."""
    addUSBOutputDeviceSignal = Signal()
    progressChanged = pyqtSignal()

    def __init__(self, application, parent=None):
        if False:
            while True:
                i = 10
        if USBPrinterOutputDeviceManager.__instance is not None:
            raise RuntimeError("Try to create singleton '%s' more than once" % self.__class__.__name__)
        super().__init__(parent=parent)
        USBPrinterOutputDeviceManager.__instance = self
        self._application = application
        self._serial_port_list = []
        self._usb_output_devices = {}
        self._usb_output_devices_model = None
        self._update_thread = threading.Thread(target=self._updateThread)
        self._update_thread.daemon = True
        self._check_updates = True
        self._application.applicationShuttingDown.connect(self.stop)
        self.addUSBOutputDeviceSignal.connect(self.addOutputDevice)
        self._application.globalContainerStackChanged.connect(self.updateUSBPrinterOutputDevices)

    def updateUSBPrinterOutputDevices(self):
        if False:
            for i in range(10):
                print('nop')
        for device in self._usb_output_devices.values():
            if isinstance(device, USBPrinterOutputDevice.USBPrinterOutputDevice):
                device.resetDeviceSettings()

    def start(self):
        if False:
            while True:
                i = 10
        self._check_updates = True
        self._update_thread.start()

    def stop(self, store_data: bool=True):
        if False:
            i = 10
            return i + 15
        self._check_updates = False

    def _onConnectionStateChanged(self, serial_port):
        if False:
            print('Hello World!')
        if serial_port not in self._usb_output_devices:
            return
        changed_device = self._usb_output_devices[serial_port]
        if changed_device.connectionState == ConnectionState.Connected:
            self.getOutputDeviceManager().addOutputDevice(changed_device)
        else:
            self.getOutputDeviceManager().removeOutputDevice(serial_port)

    def _updateThread(self):
        if False:
            print('Hello World!')
        while self._check_updates:
            container_stack = self._application.getGlobalContainerStack()
            if container_stack is None:
                time.sleep(5)
                continue
            port_list = []
            if container_stack.getMetaDataEntry('supports_usb_connection'):
                machine_file_formats = [file_type.strip() for file_type in container_stack.getMetaDataEntry('file_formats').split(';')]
                if 'text/x-gcode' in machine_file_formats:
                    port_list = self.getSerialPortList(only_list_usb=Platform.isWindows())
            self._addRemovePorts(port_list)
            time.sleep(5)

    def _addRemovePorts(self, serial_ports):
        if False:
            print('Hello World!')
        'Helper to identify serial ports (and scan for them)'
        for serial_port in list(serial_ports):
            if serial_port not in self._serial_port_list:
                self.addUSBOutputDeviceSignal.emit(serial_port)
                continue
        self._serial_port_list = list(serial_ports)
        for (port, device) in self._usb_output_devices.items():
            if port not in self._serial_port_list:
                device.close()

    def addOutputDevice(self, serial_port):
        if False:
            while True:
                i = 10
        'Because the model needs to be created in the same thread as the QMLEngine, we use a signal.'
        device = USBPrinterOutputDevice.USBPrinterOutputDevice(serial_port)
        device.connectionStateChanged.connect(self._onConnectionStateChanged)
        self._usb_output_devices[serial_port] = device
        device.connect()

    def getSerialPortList(self, only_list_usb=False):
        if False:
            while True:
                i = 10
        'Create a list of serial ports on the system.\n\n        :param only_list_usb: If true, only usb ports are listed\n        '
        base_list = []
        try:
            port_list = serial.tools.list_ports.comports()
        except TypeError:
            port_list = []
        for port in port_list:
            if not isinstance(port, tuple):
                port = (port.device, port.description, port.hwid)
            if not port[2]:
                continue
            if only_list_usb and (not port[2].startswith('USB')):
                continue
            pattern = environ.get('CURA_DEVICENAMES')
            if pattern and (not search(pattern, port[0])):
                continue
            pattern = environ.get('CURA_DEVICETYPES')
            if pattern and (not search(pattern, port[1])):
                continue
            pattern = environ.get('CURA_DEVICEINFOS')
            if pattern and (not search(pattern, port[2])):
                continue
            base_list += [port[0]]
        return list(base_list)
    __instance = None

    @classmethod
    def getInstance(cls, *args, **kwargs) -> 'USBPrinterOutputDeviceManager':
        if False:
            while True:
                i = 10
        return cls.__instance