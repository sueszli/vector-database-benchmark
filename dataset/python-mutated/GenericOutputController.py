from typing import TYPE_CHECKING, Set, Union, Optional
from PyQt6.QtCore import QTimer
from .PrinterOutputController import PrinterOutputController
if TYPE_CHECKING:
    from .Models.PrintJobOutputModel import PrintJobOutputModel
    from .Models.PrinterOutputModel import PrinterOutputModel
    from .PrinterOutputDevice import PrinterOutputDevice
    from .Models.ExtruderOutputModel import ExtruderOutputModel

class GenericOutputController(PrinterOutputController):

    def __init__(self, output_device: 'PrinterOutputDevice') -> None:
        if False:
            while True:
                i = 10
        super().__init__(output_device)
        self._preheat_bed_timer = QTimer()
        self._preheat_bed_timer.setSingleShot(True)
        self._preheat_bed_timer.timeout.connect(self._onPreheatBedTimerFinished)
        self._preheat_printer = None
        self._preheat_hotends_timer = QTimer()
        self._preheat_hotends_timer.setSingleShot(True)
        self._preheat_hotends_timer.timeout.connect(self._onPreheatHotendsTimerFinished)
        self._preheat_hotends = set()
        self._output_device.printersChanged.connect(self._onPrintersChanged)
        self._active_printer = None

    def _onPrintersChanged(self) -> None:
        if False:
            while True:
                i = 10
        if self._active_printer:
            self._active_printer.stateChanged.disconnect(self._onPrinterStateChanged)
            self._active_printer.targetBedTemperatureChanged.disconnect(self._onTargetBedTemperatureChanged)
            for extruder in self._active_printer.extruders:
                extruder.targetHotendTemperatureChanged.disconnect(self._onTargetHotendTemperatureChanged)
        self._active_printer = self._output_device.activePrinter
        if self._active_printer:
            self._active_printer.stateChanged.connect(self._onPrinterStateChanged)
            self._active_printer.targetBedTemperatureChanged.connect(self._onTargetBedTemperatureChanged)
            for extruder in self._active_printer.extruders:
                extruder.targetHotendTemperatureChanged.connect(self._onTargetHotendTemperatureChanged)

    def _onPrinterStateChanged(self) -> None:
        if False:
            while True:
                i = 10
        if self._active_printer and self._active_printer.state != 'idle':
            if self._preheat_bed_timer.isActive():
                self._preheat_bed_timer.stop()
                if self._preheat_printer:
                    self._preheat_printer.updateIsPreheating(False)
            if self._preheat_hotends_timer.isActive():
                self._preheat_hotends_timer.stop()
                for extruder in self._preheat_hotends:
                    extruder.updateIsPreheating(False)
                self._preheat_hotends = set()

    def moveHead(self, printer: 'PrinterOutputModel', x, y, z, speed) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._output_device.sendCommand('G91')
        self._output_device.sendCommand('G0 X%s Y%s Z%s F%s' % (x, y, z, speed))
        self._output_device.sendCommand('G90')

    def homeHead(self, printer: 'PrinterOutputModel') -> None:
        if False:
            return 10
        self._output_device.sendCommand('G28 X Y')

    def homeBed(self, printer: 'PrinterOutputModel') -> None:
        if False:
            while True:
                i = 10
        self._output_device.sendCommand('G28 Z')

    def sendRawCommand(self, printer: 'PrinterOutputModel', command: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._output_device.sendCommand(command.upper())

    def setJobState(self, job: 'PrintJobOutputModel', state: str) -> None:
        if False:
            i = 10
            return i + 15
        if state == 'pause':
            self._output_device.pausePrint()
            job.updateState('paused')
        elif state == 'print':
            self._output_device.resumePrint()
            job.updateState('printing')
        elif state == 'abort':
            self._output_device.cancelPrint()
            pass

    def setTargetBedTemperature(self, printer: 'PrinterOutputModel', temperature: float) -> None:
        if False:
            return 10
        self._output_device.sendCommand('M140 S%s' % round(temperature))

    def _onTargetBedTemperatureChanged(self) -> None:
        if False:
            i = 10
            return i + 15
        if self._preheat_bed_timer.isActive() and self._preheat_printer and (self._preheat_printer.targetBedTemperature == 0):
            self._preheat_bed_timer.stop()
            self._preheat_printer.updateIsPreheating(False)

    def preheatBed(self, printer: 'PrinterOutputModel', temperature, duration) -> None:
        if False:
            print('Hello World!')
        try:
            temperature = round(temperature)
            duration = round(duration)
        except ValueError:
            return
        self.setTargetBedTemperature(printer, temperature=temperature)
        self._preheat_bed_timer.setInterval(duration * 1000)
        self._preheat_bed_timer.start()
        self._preheat_printer = printer
        printer.updateIsPreheating(True)

    def cancelPreheatBed(self, printer: 'PrinterOutputModel') -> None:
        if False:
            while True:
                i = 10
        self.setTargetBedTemperature(printer, temperature=0)
        self._preheat_bed_timer.stop()
        printer.updateIsPreheating(False)

    def _onPreheatBedTimerFinished(self) -> None:
        if False:
            return 10
        if not self._preheat_printer:
            return
        self.setTargetBedTemperature(self._preheat_printer, 0)
        self._preheat_printer.updateIsPreheating(False)

    def setTargetHotendTemperature(self, printer: 'PrinterOutputModel', position: int, temperature: Union[int, float]) -> None:
        if False:
            i = 10
            return i + 15
        self._output_device.sendCommand('M104 S%s T%s' % (temperature, position))

    def _onTargetHotendTemperatureChanged(self) -> None:
        if False:
            print('Hello World!')
        if not self._preheat_hotends_timer.isActive():
            return
        if not self._active_printer:
            return
        for extruder in self._active_printer.extruders:
            if extruder in self._preheat_hotends and extruder.targetHotendTemperature == 0:
                extruder.updateIsPreheating(False)
                self._preheat_hotends.remove(extruder)
        if not self._preheat_hotends:
            self._preheat_hotends_timer.stop()

    def preheatHotend(self, extruder: 'ExtruderOutputModel', temperature, duration) -> None:
        if False:
            while True:
                i = 10
        position = extruder.getPosition()
        number_of_extruders = len(extruder.getPrinter().extruders)
        if position >= number_of_extruders:
            return
        try:
            temperature = round(temperature)
            duration = round(duration)
        except ValueError:
            return
        self.setTargetHotendTemperature(extruder.getPrinter(), position, temperature=temperature)
        self._preheat_hotends_timer.setInterval(duration * 1000)
        self._preheat_hotends_timer.start()
        self._preheat_hotends.add(extruder)
        extruder.updateIsPreheating(True)

    def cancelPreheatHotend(self, extruder: 'ExtruderOutputModel') -> None:
        if False:
            for i in range(10):
                print('nop')
        self.setTargetHotendTemperature(extruder.getPrinter(), extruder.getPosition(), temperature=0)
        if extruder in self._preheat_hotends:
            extruder.updateIsPreheating(False)
            self._preheat_hotends.remove(extruder)
        if not self._preheat_hotends and self._preheat_hotends_timer.isActive():
            self._preheat_hotends_timer.stop()

    def _onPreheatHotendsTimerFinished(self) -> None:
        if False:
            print('Hello World!')
        for extruder in self._preheat_hotends:
            self.setTargetHotendTemperature(extruder.getPrinter(), extruder.getPosition(), 0)
        self._preheat_hotends = set()

    def stopPreheatTimers(self) -> None:
        if False:
            while True:
                i = 10
        if self._preheat_hotends_timer.isActive():
            for extruder in self._preheat_hotends:
                extruder.updateIsPreheating(False)
            self._preheat_hotends = set()
            self._preheat_hotends_timer.stop()
        if self._preheat_bed_timer.isActive():
            if self._preheat_printer:
                self._preheat_printer.updateIsPreheating(False)
            self._preheat_bed_timer.stop()