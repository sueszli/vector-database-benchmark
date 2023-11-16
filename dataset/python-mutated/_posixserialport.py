"""
Serial Port Protocol
"""
from serial import PARITY_NONE
from serial import EIGHTBITS, STOPBITS_ONE
from twisted.internet import abstract, fdesc
from twisted.internet.serialport import BaseSerialPort

class SerialPort(BaseSerialPort, abstract.FileDescriptor):
    """
    A select()able serial device, acting as a transport.
    """
    connected = 1

    def __init__(self, protocol, deviceNameOrPortNumber, reactor, baudrate=9600, bytesize=EIGHTBITS, parity=PARITY_NONE, stopbits=STOPBITS_ONE, timeout=0, xonxoff=0, rtscts=0):
        if False:
            for i in range(10):
                print('nop')
        abstract.FileDescriptor.__init__(self, reactor)
        self._serial = self._serialFactory(deviceNameOrPortNumber, baudrate=baudrate, bytesize=bytesize, parity=parity, stopbits=stopbits, timeout=timeout, xonxoff=xonxoff, rtscts=rtscts)
        self.reactor = reactor
        self.flushInput()
        self.flushOutput()
        self.protocol = protocol
        self.protocol.makeConnection(self)
        self.startReading()

    def fileno(self):
        if False:
            return 10
        return self._serial.fd

    def writeSomeData(self, data):
        if False:
            while True:
                i = 10
        '\n        Write some data to the serial device.\n        '
        return fdesc.writeToFD(self.fileno(), data)

    def doRead(self):
        if False:
            print('Hello World!')
        "\n        Some data's readable from serial device.\n        "
        return fdesc.readFromFD(self.fileno(), self.protocol.dataReceived)

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when the serial port disconnects.\n\n        Will call C{connectionLost} on the protocol that is handling the\n        serial data.\n        '
        abstract.FileDescriptor.connectionLost(self, reason)
        self._serial.close()
        self.protocol.connectionLost(reason)