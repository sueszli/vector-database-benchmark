from UM.Logger import Logger
from cura.CuraApplication import CuraApplication
from cura.PrinterOutput.FirmwareUpdater import FirmwareUpdater, FirmwareUpdateState
from .avr_isp import stk500v2, intelHex
from serial import SerialException
from time import sleep
MYPY = False
if MYPY:
    from cura.PrinterOutput.PrinterOutputDevice import PrinterOutputDevice

class AvrFirmwareUpdater(FirmwareUpdater):

    def __init__(self, output_device: 'PrinterOutputDevice') -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(output_device)

    def _updateFirmware(self) -> None:
        if False:
            print('Hello World!')
        try:
            hex_file = intelHex.readHex(self._firmware_file)
            assert len(hex_file) > 0
        except (FileNotFoundError, AssertionError):
            Logger.log('e', 'Unable to read provided hex file. Could not update firmware.')
            self._setFirmwareUpdateState(FirmwareUpdateState.firmware_not_found_error)
            return
        programmer = stk500v2.Stk500v2()
        programmer.progress_callback = self._onFirmwareProgress
        if self._output_device.isConnected():
            self._output_device.close()
        try:
            programmer.connect(self._output_device._serial_port)
        except:
            programmer.close()
            Logger.logException('e', 'Failed to update firmware')
            self._setFirmwareUpdateState(FirmwareUpdateState.communication_error)
            return
        sleep(1)
        if not programmer.isConnected():
            Logger.log('e', 'Unable to connect with serial. Could not update firmware')
            self._setFirmwareUpdateState(FirmwareUpdateState.communication_error)
        try:
            programmer.programChip(hex_file)
        except SerialException as e:
            Logger.log('e', 'A serial port exception occurred during firmware update: %s' % e)
            self._setFirmwareUpdateState(FirmwareUpdateState.io_error)
            return
        except Exception as e:
            Logger.log('e', 'An unknown exception occurred during firmware update: %s' % e)
            self._setFirmwareUpdateState(FirmwareUpdateState.unknown_error)
            return
        programmer.close()
        CuraApplication.getInstance().callLater(self._output_device.connect)
        self._cleanupAfterUpdate()