"""
Tests for L{twisted.internet.serialport}.
"""
import os
import shutil
import tempfile
from twisted.internet.protocol import Protocol
from twisted.internet.test.test_serialport import DoNothing
from twisted.python.failure import Failure
from twisted.python.runtime import platform
from twisted.trial import unittest
testingForced = 'TWISTED_FORCE_SERIAL_TESTS' in os.environ
try:
    import serial
    from twisted.internet import serialport
except ImportError:
    if testingForced:
        raise
    serialport = None
    serial = None
if serialport is not None:

    class RegularFileSerial(serial.Serial):

        def __init__(self, *args, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self.captured_args = args
            self.captured_kwargs = kwargs

        def _reconfigurePort(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def _reconfigure_port(self):
            if False:
                while True:
                    i = 10
            pass

    class RegularFileSerialPort(serialport.SerialPort):
        _serialFactory = RegularFileSerial

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            cbInQue = kwargs.get('cbInQue')
            if 'cbInQue' in kwargs:
                del kwargs['cbInQue']
            self.comstat = serial.win32.COMSTAT
            self.comstat.cbInQue = cbInQue
            super().__init__(*args, **kwargs)

        def _clearCommError(self):
            if False:
                while True:
                    i = 10
            return (True, self.comstat)

class CollectReceivedProtocol(Protocol):

    def __init__(self):
        if False:
            return 10
        self.received_data = []

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.received_data.append(data)

class Win32SerialPortTests(unittest.TestCase):
    """
    Minimal testing for Twisted's Win32 serial port support.
    """
    if not testingForced:
        if not platform.isWindows():
            skip = 'This test must run on Windows.'
        elif not serialport:
            skip = 'Windows serial port support is not available.'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.protocol = Protocol()
        self.reactor = DoNothing()
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, 'fake_serial')
        data = b'1234'
        with open(self.path, 'wb') as f:
            f.write(data)

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.directory)

    def test_serialPortDefaultArgs(self):
        if False:
            while True:
                i = 10
        '\n        Test correct positional and keyword arguments have been\n        passed to the C{serial.Serial} object.\n        '
        port = RegularFileSerialPort(self.protocol, self.path, self.reactor)
        self.assertEqual((self.path,), port._serial.captured_args)
        kwargs = port._serial.captured_kwargs
        self.assertEqual(9600, kwargs['baudrate'])
        self.assertEqual(serial.EIGHTBITS, kwargs['bytesize'])
        self.assertEqual(serial.PARITY_NONE, kwargs['parity'])
        self.assertEqual(serial.STOPBITS_ONE, kwargs['stopbits'])
        self.assertEqual(0, kwargs['xonxoff'])
        self.assertEqual(0, kwargs['rtscts'])
        self.assertEqual(None, kwargs['timeout'])
        port.connectionLost(Failure(Exception('Cleanup')))

    def test_serialPortInitiallyConnected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the port is connected at initialization time, and\n        C{Protocol.makeConnection} has been called on the desired protocol.\n        '
        self.assertEqual(0, self.protocol.connected)
        port = RegularFileSerialPort(self.protocol, self.path, self.reactor)
        self.assertEqual(1, port.connected)
        self.assertEqual(1, self.protocol.connected)
        self.assertEqual(port, self.protocol.transport)
        port.connectionLost(Failure(Exception('Cleanup')))

    def common_exerciseHandleAccess(self, cbInQue):
        if False:
            i = 10
            return i + 15
        port = RegularFileSerialPort(protocol=self.protocol, deviceNameOrPortNumber=self.path, reactor=self.reactor, cbInQue=cbInQue)
        port.serialReadEvent()
        port.write(b'')
        port.write(b'abcd')
        port.write(b'ABCD')
        port.serialWriteEvent()
        port.serialWriteEvent()
        port.connectionLost(Failure(Exception('Cleanup')))

    def test_exerciseHandleAccess_1(self):
        if False:
            while True:
                i = 10
        self.common_exerciseHandleAccess(cbInQue=False)

    def test_exerciseHandleAccess_2(self):
        if False:
            while True:
                i = 10
        self.common_exerciseHandleAccess(cbInQue=True)

    def common_serialPortReturnsBytes(self, cbInQue):
        if False:
            while True:
                i = 10
        protocol = CollectReceivedProtocol()
        port = RegularFileSerialPort(protocol=protocol, deviceNameOrPortNumber=self.path, reactor=self.reactor, cbInQue=cbInQue)
        port.serialReadEvent()
        self.assertTrue(all((isinstance(d, bytes) for d in protocol.received_data)))
        port.connectionLost(Failure(Exception('Cleanup')))

    def test_serialPortReturnsBytes_1(self):
        if False:
            i = 10
            return i + 15
        self.common_serialPortReturnsBytes(cbInQue=False)

    def test_serialPortReturnsBytes_2(self):
        if False:
            for i in range(10):
                print('nop')
        self.common_serialPortReturnsBytes(cbInQue=True)