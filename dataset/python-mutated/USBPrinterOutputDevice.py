import os
from UM.i18n import i18nCatalog
from UM.Logger import Logger
from UM.Mesh.MeshWriter import MeshWriter
from UM.Message import Message
from UM.PluginRegistry import PluginRegistry
from UM.Qt.Duration import DurationFormat
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.PrinterOutputDevice import PrinterOutputDevice, ConnectionState, ConnectionType
from cura.PrinterOutput.Models.PrinterOutputModel import PrinterOutputModel
from cura.PrinterOutput.Models.PrintJobOutputModel import PrintJobOutputModel
from cura.PrinterOutput.GenericOutputController import GenericOutputController
from .AutoDetectBaudJob import AutoDetectBaudJob
from .AvrFirmwareUpdater import AvrFirmwareUpdater
from io import StringIO
from queue import Queue
from serial import Serial, SerialException, SerialTimeoutException
from threading import Thread, Event
from time import time
from typing import Union, Optional, List, cast, TYPE_CHECKING
import re
import functools
if TYPE_CHECKING:
    from UM.FileHandler.FileHandler import FileHandler
    from UM.Scene.SceneNode import SceneNode
catalog = i18nCatalog('cura')

class USBPrinterOutputDevice(PrinterOutputDevice):

    def __init__(self, serial_port: str, baud_rate: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(serial_port, connection_type=ConnectionType.UsbConnection)
        self.setName(catalog.i18nc('@item:inmenu', 'USB printing'))
        self.setShortDescription(catalog.i18nc("@action:button Preceded by 'Ready to'.", 'Print via USB'))
        self.setDescription(catalog.i18nc('@info:tooltip', 'Print via USB'))
        self.setIconName('print')
        self._serial = None
        self._serial_port = serial_port
        self._address = serial_port
        self._timeout = 3
        self._gcode = []
        self._gcode_position = 0
        self._use_auto_detect = True
        self._baud_rate = baud_rate
        self._all_baud_rates = [115200, 250000, 500000, 230400, 76800, 57600, 38400, 19200, 9600]
        self._update_thread = Thread(target=self._update, daemon=True, name='USBPrinterUpdate')
        self._last_temperature_request = None
        self._firmware_idle_count = 0
        self._is_printing = False
        self._print_start_time = None
        self._print_estimated_time = None
        self._accepts_commands = True
        self._paused = False
        self._printer_busy = False
        self.setConnectionText(catalog.i18nc('@info:status', 'Connected via USB'))
        self._command_queue = Queue()
        self._command_received = Event()
        self._command_received.set()
        self._firmware_name_requested = False
        self._firmware_updater = AvrFirmwareUpdater(self)
        plugin_path = PluginRegistry.getInstance().getPluginPath('USBPrinting')
        if plugin_path:
            self._monitor_view_qml_path = os.path.join(plugin_path, 'MonitorItem.qml')
        else:
            Logger.log('e', 'Cannot create Monitor QML view: cannot find plugin path for plugin [USBPrinting]')
            self._monitor_view_qml_path = ''
        CuraApplication.getInstance().getOnExitCallbackManager().addCallback(self._checkActivePrintingUponAppExit)

    def _checkActivePrintingUponAppExit(self) -> None:
        if False:
            while True:
                i = 10
        application = CuraApplication.getInstance()
        if not self._is_printing:
            application.triggerNextExitCheck()
            return
        application.setConfirmExitDialogCallback(self._onConfirmExitDialogResult)
        application.showConfirmExitDialog.emit(catalog.i18nc('@label', 'A USB print is in progress, closing Cura will stop this print. Are you sure?'))

    def _onConfirmExitDialogResult(self, result: bool) -> None:
        if False:
            i = 10
            return i + 15
        if result:
            application = CuraApplication.getInstance()
            application.triggerNextExitCheck()

    def resetDeviceSettings(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Reset USB device settings'
        self._firmware_name = None

    def requestWrite(self, nodes: List['SceneNode'], file_name: Optional[str]=None, limit_mimetypes: bool=False, file_handler: Optional['FileHandler']=None, filter_by_machine: bool=False, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        'Request the current scene to be sent to a USB-connected printer.\n\n        :param nodes: A collection of scene nodes to send. This is ignored.\n        :param file_name: A suggestion for a file name to write.\n        :param filter_by_machine: Whether to filter MIME types by machine. This\n               is ignored.\n        :param kwargs: Keyword arguments.\n        '
        if self._is_printing:
            message = Message(text=catalog.i18nc('@message', 'A print is still in progress. Cura cannot start another print via USB until the previous print has completed.'), title=catalog.i18nc('@message', 'Print in Progress'), message_type=Message.MessageType.ERROR)
            message.show()
            return
        self.writeStarted.emit(self)
        controller = cast(GenericOutputController, self._printers[0].getController())
        controller.stopPreheatTimers()
        CuraApplication.getInstance().getController().setActiveStage('MonitorStage')
        gcode_textio = StringIO()
        gcode_writer = cast(MeshWriter, PluginRegistry.getInstance().getPluginObject('GCodeWriter'))
        success = gcode_writer.write(gcode_textio, None)
        if not success:
            return
        self._printGCode(gcode_textio.getvalue())

    def _printGCode(self, gcode: str):
        if False:
            while True:
                i = 10
        'Start a print based on a g-code.\n\n        :param gcode: The g-code to print.\n        '
        self._gcode.clear()
        self._paused = False
        self._gcode.extend(gcode.split('\n'))
        self._gcode.insert(0, 'M110')
        self._gcode_position = 0
        self._print_start_time = time()
        self._print_estimated_time = int(CuraApplication.getInstance().getPrintInformation().currentPrintTime.getDisplayString(DurationFormat.Format.Seconds))
        for i in range(0, 4):
            self._sendNextGcodeLine()
        self._is_printing = True
        self.writeFinished.emit(self)

    def _autoDetectFinished(self, job: AutoDetectBaudJob):
        if False:
            while True:
                i = 10
        result = job.getResult()
        if result is not None:
            self.setBaudRate(result)
            self.connect()

    def setBaudRate(self, baud_rate: int):
        if False:
            i = 10
            return i + 15
        if baud_rate not in self._all_baud_rates:
            Logger.log('w', "Not updating baudrate to {baud_rate} as it's an unknown baudrate".format(baud_rate=baud_rate))
            return
        self._baud_rate = baud_rate

    def connect(self):
        if False:
            print('Hello World!')
        self._firmware_name = None
        if self._baud_rate is None:
            if self._use_auto_detect:
                auto_detect_job = AutoDetectBaudJob(self._serial_port)
                auto_detect_job.start()
                auto_detect_job.finished.connect(self._autoDetectFinished)
            return
        if self._serial is None:
            try:
                self._serial = Serial(str(self._serial_port), self._baud_rate, timeout=self._timeout, writeTimeout=self._timeout)
            except SerialException:
                Logger.warning('An exception occurred while trying to create serial connection.')
                return
            except OSError as e:
                Logger.warning('The serial device is suddenly unavailable while trying to create a serial connection: {err}'.format(err=str(e)))
                return
        CuraApplication.getInstance().globalContainerStackChanged.connect(self._onGlobalContainerStackChanged)
        self._onGlobalContainerStackChanged()
        self.setConnectionState(ConnectionState.Connected)
        self._update_thread.start()

    def _onGlobalContainerStackChanged(self):
        if False:
            while True:
                i = 10
        container_stack = CuraApplication.getInstance().getGlobalContainerStack()
        if container_stack is None:
            return
        num_extruders = container_stack.getProperty('machine_extruder_count', 'value')
        controller = GenericOutputController(self)
        controller.setCanUpdateFirmware(True)
        self._printers = [PrinterOutputModel(output_controller=controller, number_of_extruders=num_extruders)]
        self._printers[0].updateName(container_stack.getName())

    def close(self):
        if False:
            while True:
                i = 10
        super().close()
        if self._serial is not None:
            self._serial.close()
        self._update_thread = Thread(target=self._update, daemon=True, name='USBPrinterUpdate')
        self._serial = None

    def sendCommand(self, command: Union[str, bytes]):
        if False:
            for i in range(10):
                print('nop')
        'Send a command to printer.'
        if not self._command_received.is_set():
            self._command_queue.put(command)
        else:
            self._sendCommand(command)

    def _sendCommand(self, command: Union[str, bytes]):
        if False:
            i = 10
            return i + 15
        if self._serial is None or self._connection_state != ConnectionState.Connected:
            return
        new_command = cast(bytes, command) if type(command) is bytes else cast(str, command).encode()
        if not new_command.endswith(b'\n'):
            new_command += b'\n'
        try:
            self._command_received.clear()
            self._serial.write(new_command)
        except SerialTimeoutException:
            Logger.log('w', 'Timeout when sending command to printer via USB.')
            self._command_received.set()
        except SerialException:
            Logger.logException('w', 'An unexpected exception occurred while writing to the serial.')
            self.setConnectionState(ConnectionState.Error)

    def _update(self):
        if False:
            i = 10
            return i + 15
        while self._connection_state == ConnectionState.Connected and self._serial is not None:
            try:
                line = self._serial.readline()
            except:
                continue
            if not self._firmware_name_requested:
                self._firmware_name_requested = True
                self.sendCommand('M115')
            if b'FIRMWARE_NAME:' in line:
                self._setFirmwareName(line)
            if self._last_temperature_request is None or time() > self._last_temperature_request + self._timeout:
                if not self._printer_busy:
                    self.sendCommand('M105')
                    self._last_temperature_request = time()
            if re.search(b'[B|T\\d*]: ?\\d+\\.?\\d*', line):
                extruder_temperature_matches = re.findall(b'T(\\d*): ?(\\d+\\.?\\d*)\\s*\\/?(\\d+\\.?\\d*)?', line)
                matched_extruder_nrs = []
                for match in extruder_temperature_matches:
                    extruder_nr = 0
                    if match[0] != b'':
                        extruder_nr = int(match[0])
                    if extruder_nr in matched_extruder_nrs:
                        continue
                    matched_extruder_nrs.append(extruder_nr)
                    if extruder_nr >= len(self._printers[0].extruders):
                        Logger.log('w', 'Printer reports more temperatures than the number of configured extruders')
                        continue
                    extruder = self._printers[0].extruders[extruder_nr]
                    if match[1]:
                        extruder.updateHotendTemperature(float(match[1]))
                    if match[2]:
                        extruder.updateTargetHotendTemperature(float(match[2]))
                bed_temperature_matches = re.findall(b'B: ?(\\d+\\.?\\d*)\\s*\\/?(\\d+\\.?\\d*)?', line)
                if bed_temperature_matches:
                    match = bed_temperature_matches[0]
                    if match[0]:
                        self._printers[0].updateBedTemperature(float(match[0]))
                    if match[1]:
                        self._printers[0].updateTargetBedTemperature(float(match[1]))
            if line == b'':
                self._firmware_idle_count += 1
            else:
                self._firmware_idle_count = 0
            if line.startswith(b'ok') or self._firmware_idle_count > 1:
                self._printer_busy = False
                self._command_received.set()
                if not self._command_queue.empty():
                    self._sendCommand(self._command_queue.get())
                elif self._is_printing:
                    if self._paused:
                        pass
                    else:
                        self._sendNextGcodeLine()
            if line.startswith(b'echo:busy:'):
                self._printer_busy = True
            if self._is_printing:
                if line.startswith(b'!!'):
                    Logger.log('e', 'Printer signals fatal error. Cancelling print. {}'.format(line))
                    self.cancelPrint()
                elif line.lower().startswith(b'resend') or line.startswith(b'rs'):
                    try:
                        self._gcode_position = int(line.replace(b'N:', b' ').replace(b'N', b' ').replace(b':', b' ').split()[-1])
                    except:
                        if line.startswith(b'rs'):
                            self._gcode_position = int(line.split()[1])

    def _setFirmwareName(self, name):
        if False:
            i = 10
            return i + 15
        new_name = re.findall('FIRMWARE_NAME:(.*);', str(name))
        if new_name:
            self._firmware_name = new_name[0]
            Logger.log('i', 'USB output device Firmware name: %s', self._firmware_name)
        else:
            self._firmware_name = 'Unknown'
            Logger.log('i', 'Unknown USB output device Firmware name: %s', str(name))

    def getFirmwareName(self):
        if False:
            for i in range(10):
                print('nop')
        return self._firmware_name

    def pausePrint(self):
        if False:
            return 10
        self._paused = True

    def resumePrint(self):
        if False:
            while True:
                i = 10
        self._paused = False
        self._sendNextGcodeLine()

    def cancelPrint(self):
        if False:
            while True:
                i = 10
        self._gcode_position = 0
        self._gcode.clear()
        self._printers[0].updateActivePrintJob(None)
        self._is_printing = False
        self._paused = False
        self._sendCommand('M140 S0')
        self._sendCommand('M104 S0')
        self._sendCommand('M107')
        self.printers[0].homeHead()
        self._sendCommand('M84')

    def _sendNextGcodeLine(self):
        if False:
            return 10
        '\n        Send the next line of g-code, at the current `_gcode_position`, via a\n        serial port to the printer.\n\n        If the print is done, this sets `_is_printing` to `False` as well.\n        '
        try:
            line = self._gcode[self._gcode_position]
        except IndexError:
            self._printers[0].updateActivePrintJob(None)
            self._is_printing = False
            return
        if ';' in line:
            line = line[:line.find(';')]
        line = line.strip()
        if line == '' or line == 'M0' or line == 'M1':
            line = 'M105'
        checksum = functools.reduce(lambda x, y: x ^ y, map(ord, 'N%d%s' % (self._gcode_position, line)))
        self._sendCommand('N%d%s*%d' % (self._gcode_position, line, checksum))
        print_job = self._printers[0].activePrintJob
        try:
            progress = self._gcode_position / len(self._gcode)
        except ZeroDivisionError:
            if print_job is not None:
                print_job.updateState('error')
            return
        elapsed_time = int(time() - self._print_start_time)
        if print_job is None:
            controller = GenericOutputController(self)
            controller.setCanUpdateFirmware(True)
            print_job = PrintJobOutputModel(output_controller=controller, name=CuraApplication.getInstance().getPrintInformation().jobName)
            print_job.updateState('printing')
            self._printers[0].updateActivePrintJob(print_job)
        print_job.updateTimeElapsed(elapsed_time)
        estimated_time = self._print_estimated_time
        if progress > 0.1:
            estimated_time = int(self._print_estimated_time * (1 - progress) + elapsed_time)
        print_job.updateTimeTotal(estimated_time)
        self._gcode_position += 1