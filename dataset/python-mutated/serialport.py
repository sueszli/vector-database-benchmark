"""
Serial Port Protocol
"""
__all__ = ['serial', 'PARITY_ODD', 'PARITY_EVEN', 'PARITY_NONE', 'STOPBITS_TWO', 'STOPBITS_ONE', 'FIVEBITS', 'EIGHTBITS', 'SEVENBITS', 'SIXBITS', 'SerialPort']
import serial
from serial import EIGHTBITS, FIVEBITS, PARITY_EVEN, PARITY_NONE, PARITY_ODD, SEVENBITS, SIXBITS, STOPBITS_ONE, STOPBITS_TWO
from twisted.python.runtime import platform

class BaseSerialPort:
    """
    Base class for Windows and POSIX serial ports.

    @ivar _serialFactory: a pyserial C{serial.Serial} factory, used to create
        the instance stored in C{self._serial}. Overrideable to enable easier
        testing.

    @ivar _serial: a pyserial C{serial.Serial} instance used to manage the
        options on the serial port.
    """
    _serialFactory = serial.Serial

    def setBaudRate(self, baudrate):
        if False:
            print('Hello World!')
        if hasattr(self._serial, 'setBaudrate'):
            self._serial.setBaudrate(baudrate)
        else:
            self._serial.setBaudRate(baudrate)

    def inWaiting(self):
        if False:
            while True:
                i = 10
        return self._serial.inWaiting()

    def flushInput(self):
        if False:
            i = 10
            return i + 15
        self._serial.flushInput()

    def flushOutput(self):
        if False:
            i = 10
            return i + 15
        self._serial.flushOutput()

    def sendBreak(self):
        if False:
            print('Hello World!')
        self._serial.sendBreak()

    def getDSR(self):
        if False:
            while True:
                i = 10
        return self._serial.getDSR()

    def getCD(self):
        if False:
            while True:
                i = 10
        return self._serial.getCD()

    def getRI(self):
        if False:
            return 10
        return self._serial.getRI()

    def getCTS(self):
        if False:
            return 10
        return self._serial.getCTS()

    def setDTR(self, on=1):
        if False:
            print('Hello World!')
        self._serial.setDTR(on)

    def setRTS(self, on=1):
        if False:
            return 10
        self._serial.setRTS(on)
if platform.isWindows():
    from twisted.internet._win32serialport import SerialPort
else:
    from twisted.internet._posixserialport import SerialPort