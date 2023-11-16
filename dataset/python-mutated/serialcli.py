from __future__ import absolute_import
import System
import System.IO.Ports
from serial.serialutil import *
sab = System.Array[System.Byte]

def as_byte_array(string):
    if False:
        print('Hello World!')
    return sab([ord(x) for x in string])

class Serial(SerialBase):
    """Serial port implementation for .NET/Mono."""
    BAUDRATES = (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200)

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        '        Open port with current settings. This may throw a SerialException\n        if the port cannot be opened.\n        '
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if self.is_open:
            raise SerialException('Port is already open.')
        try:
            self._port_handle = System.IO.Ports.SerialPort(self.portstr)
        except Exception as msg:
            self._port_handle = None
            raise SerialException('could not open port %s: %s' % (self.portstr, msg))
        if self._rts_state is None:
            self._rts_state = True
        if self._dtr_state is None:
            self._dtr_state = True
        self._reconfigure_port()
        self._port_handle.Open()
        self.is_open = True
        if not self._dsrdtr:
            self._update_dtr_state()
        if not self._rtscts:
            self._update_rts_state()
        self.reset_input_buffer()

    def _reconfigure_port(self):
        if False:
            while True:
                i = 10
        'Set communication parameters on opened port.'
        if not self._port_handle:
            raise SerialException('Can only operate on a valid port handle')
        if self._timeout is None:
            self._port_handle.ReadTimeout = System.IO.Ports.SerialPort.InfiniteTimeout
        else:
            self._port_handle.ReadTimeout = int(self._timeout * 1000)
        if self._write_timeout is None:
            self._port_handle.WriteTimeout = System.IO.Ports.SerialPort.InfiniteTimeout
        else:
            self._port_handle.WriteTimeout = int(self._write_timeout * 1000)
        try:
            self._port_handle.BaudRate = self._baudrate
        except IOError as e:
            raise ValueError(str(e))
        if self._bytesize == FIVEBITS:
            self._port_handle.DataBits = 5
        elif self._bytesize == SIXBITS:
            self._port_handle.DataBits = 6
        elif self._bytesize == SEVENBITS:
            self._port_handle.DataBits = 7
        elif self._bytesize == EIGHTBITS:
            self._port_handle.DataBits = 8
        else:
            raise ValueError('Unsupported number of data bits: %r' % self._bytesize)
        if self._parity == PARITY_NONE:
            self._port_handle.Parity = getattr(System.IO.Ports.Parity, 'None')
        elif self._parity == PARITY_EVEN:
            self._port_handle.Parity = System.IO.Ports.Parity.Even
        elif self._parity == PARITY_ODD:
            self._port_handle.Parity = System.IO.Ports.Parity.Odd
        elif self._parity == PARITY_MARK:
            self._port_handle.Parity = System.IO.Ports.Parity.Mark
        elif self._parity == PARITY_SPACE:
            self._port_handle.Parity = System.IO.Ports.Parity.Space
        else:
            raise ValueError('Unsupported parity mode: %r' % self._parity)
        if self._stopbits == STOPBITS_ONE:
            self._port_handle.StopBits = System.IO.Ports.StopBits.One
        elif self._stopbits == STOPBITS_ONE_POINT_FIVE:
            self._port_handle.StopBits = System.IO.Ports.StopBits.OnePointFive
        elif self._stopbits == STOPBITS_TWO:
            self._port_handle.StopBits = System.IO.Ports.StopBits.Two
        else:
            raise ValueError('Unsupported number of stop bits: %r' % self._stopbits)
        if self._rtscts and self._xonxoff:
            self._port_handle.Handshake = System.IO.Ports.Handshake.RequestToSendXOnXOff
        elif self._rtscts:
            self._port_handle.Handshake = System.IO.Ports.Handshake.RequestToSend
        elif self._xonxoff:
            self._port_handle.Handshake = System.IO.Ports.Handshake.XOnXOff
        else:
            self._port_handle.Handshake = getattr(System.IO.Ports.Handshake, 'None')

    def close(self):
        if False:
            while True:
                i = 10
        'Close port'
        if self.is_open:
            if self._port_handle:
                try:
                    self._port_handle.Close()
                except System.IO.Ports.InvalidOperationException:
                    pass
                self._port_handle = None
            self.is_open = False

    @property
    def in_waiting(self):
        if False:
            i = 10
            return i + 15
        'Return the number of characters currently in the input buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        return self._port_handle.BytesToRead

    def read(self, size=1):
        if False:
            print('Hello World!')
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        data = bytearray()
        while size:
            try:
                data.append(self._port_handle.ReadByte())
            except System.TimeoutException:
                break
            else:
                size -= 1
        return bytes(data)

    def write(self, data):
        if False:
            return 10
        'Output the given string over the serial port.'
        if not self.is_open:
            raise PortNotOpenError()
        try:
            self._port_handle.Write(as_byte_array(data), 0, len(data))
        except System.TimeoutException:
            raise SerialTimeoutException('Write timeout')
        return len(data)

    def reset_input_buffer(self):
        if False:
            print('Hello World!')
        'Clear input buffer, discarding all that is in the buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        self._port_handle.DiscardInBuffer()

    def reset_output_buffer(self):
        if False:
            return 10
        '        Clear output buffer, aborting the current output and\n        discarding all that is in the buffer.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        self._port_handle.DiscardOutBuffer()

    def _update_break_state(self):
        if False:
            return 10
        '\n        Set break: Controls TXD. When active, to transmitting is possible.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        self._port_handle.BreakState = bool(self._break_state)

    def _update_rts_state(self):
        if False:
            while True:
                i = 10
        'Set terminal status line: Request To Send'
        if not self.is_open:
            raise PortNotOpenError()
        self._port_handle.RtsEnable = bool(self._rts_state)

    def _update_dtr_state(self):
        if False:
            i = 10
            return i + 15
        'Set terminal status line: Data Terminal Ready'
        if not self.is_open:
            raise PortNotOpenError()
        self._port_handle.DtrEnable = bool(self._dtr_state)

    @property
    def cts(self):
        if False:
            for i in range(10):
                print('nop')
        'Read terminal status line: Clear To Send'
        if not self.is_open:
            raise PortNotOpenError()
        return self._port_handle.CtsHolding

    @property
    def dsr(self):
        if False:
            for i in range(10):
                print('nop')
        'Read terminal status line: Data Set Ready'
        if not self.is_open:
            raise PortNotOpenError()
        return self._port_handle.DsrHolding

    @property
    def ri(self):
        if False:
            for i in range(10):
                print('nop')
        'Read terminal status line: Ring Indicator'
        if not self.is_open:
            raise PortNotOpenError()
        return False

    @property
    def cd(self):
        if False:
            return 10
        'Read terminal status line: Carrier Detect'
        if not self.is_open:
            raise PortNotOpenError()
        return self._port_handle.CDHolding