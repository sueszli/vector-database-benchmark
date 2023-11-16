from UM.Logger import Logger
from UM.Signal import Signal
MYPY = False
if MYPY:
    from .Models.PrintJobOutputModel import PrintJobOutputModel
    from .Models.ExtruderOutputModel import ExtruderOutputModel
    from .Models.PrinterOutputModel import PrinterOutputModel
    from .PrinterOutputDevice import PrinterOutputDevice

class PrinterOutputController:

    def __init__(self, output_device: 'PrinterOutputDevice') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.can_pause = True
        self.can_abort = True
        self.can_pre_heat_bed = True
        self.can_pre_heat_hotends = True
        self.can_send_raw_gcode = True
        self.can_control_manually = True
        self.can_update_firmware = False
        self._output_device = output_device

    def setTargetHotendTemperature(self, printer: 'PrinterOutputModel', position: int, temperature: float) -> None:
        if False:
            while True:
                i = 10
        Logger.log('w', 'Set target hotend temperature not implemented in controller')

    def setTargetBedTemperature(self, printer: 'PrinterOutputModel', temperature: float) -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('w', 'Set target bed temperature not implemented in controller')

    def setJobState(self, job: 'PrintJobOutputModel', state: str) -> None:
        if False:
            print('Hello World!')
        Logger.log('w', 'Set job state not implemented in controller')

    def cancelPreheatBed(self, printer: 'PrinterOutputModel') -> None:
        if False:
            return 10
        Logger.log('w', 'Cancel preheat bed not implemented in controller')

    def preheatBed(self, printer: 'PrinterOutputModel', temperature, duration) -> None:
        if False:
            print('Hello World!')
        Logger.log('w', 'Preheat bed not implemented in controller')

    def cancelPreheatHotend(self, extruder: 'ExtruderOutputModel') -> None:
        if False:
            for i in range(10):
                print('nop')
        Logger.log('w', 'Cancel preheat hotend not implemented in controller')

    def preheatHotend(self, extruder: 'ExtruderOutputModel', temperature, duration) -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('w', 'Preheat hotend not implemented in controller')

    def setHeadPosition(self, printer: 'PrinterOutputModel', x, y, z, speed) -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('w', 'Set head position not implemented in controller')

    def moveHead(self, printer: 'PrinterOutputModel', x, y, z, speed) -> None:
        if False:
            for i in range(10):
                print('nop')
        Logger.log('w', 'Move head not implemented in controller')

    def homeBed(self, printer: 'PrinterOutputModel') -> None:
        if False:
            i = 10
            return i + 15
        Logger.log('w', 'Home bed not implemented in controller')

    def homeHead(self, printer: 'PrinterOutputModel') -> None:
        if False:
            for i in range(10):
                print('nop')
        Logger.log('w', 'Home head not implemented in controller')

    def sendRawCommand(self, printer: 'PrinterOutputModel', command: str) -> None:
        if False:
            return 10
        Logger.log('w', 'Custom command not implemented in controller')
    canUpdateFirmwareChanged = Signal()

    def setCanUpdateFirmware(self, can_update_firmware: bool) -> None:
        if False:
            print('Hello World!')
        if can_update_firmware != self.can_update_firmware:
            self.can_update_firmware = can_update_firmware
            self.canUpdateFirmwareChanged.emit()