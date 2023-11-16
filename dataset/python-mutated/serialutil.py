from __future__ import absolute_import
import io
import time
try:
    memoryview
except (NameError, AttributeError):

    class memoryview(object):
        pass
try:
    unicode
except (NameError, AttributeError):
    unicode = str
try:
    basestring
except (NameError, AttributeError):
    basestring = (str,)

def iterbytes(b):
    if False:
        return 10
    'Iterate over bytes, returning bytes instead of ints (python3)'
    if isinstance(b, memoryview):
        b = b.tobytes()
    i = 0
    while True:
        a = b[i:i + 1]
        i += 1
        if a:
            yield a
        else:
            break

def to_bytes(seq):
    if False:
        return 10
    'convert a sequence to a bytes type'
    if isinstance(seq, bytes):
        return seq
    elif isinstance(seq, bytearray):
        return bytes(seq)
    elif isinstance(seq, memoryview):
        return seq.tobytes()
    elif isinstance(seq, unicode):
        raise TypeError('unicode strings are not supported, please encode to bytes: {!r}'.format(seq))
    else:
        return bytes(bytearray(seq))
XON = to_bytes([17])
XOFF = to_bytes([19])
CR = to_bytes([13])
LF = to_bytes([10])
(PARITY_NONE, PARITY_EVEN, PARITY_ODD, PARITY_MARK, PARITY_SPACE) = ('N', 'E', 'O', 'M', 'S')
(STOPBITS_ONE, STOPBITS_ONE_POINT_FIVE, STOPBITS_TWO) = (1, 1.5, 2)
(FIVEBITS, SIXBITS, SEVENBITS, EIGHTBITS) = (5, 6, 7, 8)
PARITY_NAMES = {PARITY_NONE: 'None', PARITY_EVEN: 'Even', PARITY_ODD: 'Odd', PARITY_MARK: 'Mark', PARITY_SPACE: 'Space'}

class SerialException(IOError):
    """Base class for serial port related exceptions."""

class SerialTimeoutException(SerialException):
    """Write timeouts give an exception"""

class PortNotOpenError(SerialException):
    """Port is not open"""

    def __init__(self):
        if False:
            print('Hello World!')
        super(PortNotOpenError, self).__init__('Attempting to use a port that is not open')

class Timeout(object):
    """    Abstraction for timeout operations. Using time.monotonic() if available
    or time.time() in all other cases.

    The class can also be initialized with 0 or None, in order to support
    non-blocking and fully blocking I/O operations. The attributes
    is_non_blocking and is_infinite are set accordingly.
    """
    if hasattr(time, 'monotonic'):
        TIME = time.monotonic
    else:
        TIME = time.time

    def __init__(self, duration):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a timeout with given duration'
        self.is_infinite = duration is None
        self.is_non_blocking = duration == 0
        self.duration = duration
        if duration is not None:
            self.target_time = self.TIME() + duration
        else:
            self.target_time = None

    def expired(self):
        if False:
            i = 10
            return i + 15
        'Return a boolean, telling if the timeout has expired'
        return self.target_time is not None and self.time_left() <= 0

    def time_left(self):
        if False:
            for i in range(10):
                print('nop')
        'Return how many seconds are left until the timeout expires'
        if self.is_non_blocking:
            return 0
        elif self.is_infinite:
            return None
        else:
            delta = self.target_time - self.TIME()
            if delta > self.duration:
                self.target_time = self.TIME() + self.duration
                return self.duration
            else:
                return max(0, delta)

    def restart(self, duration):
        if False:
            while True:
                i = 10
        '        Restart a timeout, only supported if a timeout was already set up\n        before.\n        '
        self.duration = duration
        self.target_time = self.TIME() + duration

class SerialBase(io.RawIOBase):
    """    Serial port base class. Provides __init__ function and properties to
    get/set port settings.
    """
    BAUDRATES = (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000, 576000, 921600, 1000000, 1152000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000)
    BYTESIZES = (FIVEBITS, SIXBITS, SEVENBITS, EIGHTBITS)
    PARITIES = (PARITY_NONE, PARITY_EVEN, PARITY_ODD, PARITY_MARK, PARITY_SPACE)
    STOPBITS = (STOPBITS_ONE, STOPBITS_ONE_POINT_FIVE, STOPBITS_TWO)

    def __init__(self, port=None, baudrate=9600, bytesize=EIGHTBITS, parity=PARITY_NONE, stopbits=STOPBITS_ONE, timeout=None, xonxoff=False, rtscts=False, write_timeout=None, dsrdtr=False, inter_byte_timeout=None, exclusive=None, **kwargs):
        if False:
            return 10
        '        Initialize comm port object. If a "port" is given, then the port will be\n        opened immediately. Otherwise a Serial port object in closed state\n        is returned.\n        '
        self.is_open = False
        self.portstr = None
        self.name = None
        self._port = None
        self._baudrate = None
        self._bytesize = None
        self._parity = None
        self._stopbits = None
        self._timeout = None
        self._write_timeout = None
        self._xonxoff = None
        self._rtscts = None
        self._dsrdtr = None
        self._inter_byte_timeout = None
        self._rs485_mode = None
        self._rts_state = True
        self._dtr_state = True
        self._break_state = False
        self._exclusive = None
        self.port = port
        self.baudrate = baudrate
        self.bytesize = bytesize
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.xonxoff = xonxoff
        self.rtscts = rtscts
        self.dsrdtr = dsrdtr
        self.inter_byte_timeout = inter_byte_timeout
        self.exclusive = exclusive
        if 'writeTimeout' in kwargs:
            self.write_timeout = kwargs.pop('writeTimeout')
        if 'interCharTimeout' in kwargs:
            self.inter_byte_timeout = kwargs.pop('interCharTimeout')
        if kwargs:
            raise ValueError('unexpected keyword arguments: {!r}'.format(kwargs))
        if port is not None:
            self.open()

    @property
    def port(self):
        if False:
            print('Hello World!')
        '        Get the current port setting. The value that was passed on init or using\n        setPort() is passed back.\n        '
        return self._port

    @port.setter
    def port(self, port):
        if False:
            while True:
                i = 10
        '        Change the port.\n        '
        if port is not None and (not isinstance(port, basestring)):
            raise ValueError('"port" must be None or a string, not {}'.format(type(port)))
        was_open = self.is_open
        if was_open:
            self.close()
        self.portstr = port
        self._port = port
        self.name = self.portstr
        if was_open:
            self.open()

    @property
    def baudrate(self):
        if False:
            i = 10
            return i + 15
        'Get the current baud rate setting.'
        return self._baudrate

    @baudrate.setter
    def baudrate(self, baudrate):
        if False:
            while True:
                i = 10
        '        Change baud rate. It raises a ValueError if the port is open and the\n        baud rate is not possible. If the port is closed, then the value is\n        accepted and the exception is raised when the port is opened.\n        '
        try:
            b = int(baudrate)
        except TypeError:
            raise ValueError('Not a valid baudrate: {!r}'.format(baudrate))
        else:
            if b < 0:
                raise ValueError('Not a valid baudrate: {!r}'.format(baudrate))
            self._baudrate = b
            if self.is_open:
                self._reconfigure_port()

    @property
    def bytesize(self):
        if False:
            while True:
                i = 10
        'Get the current byte size setting.'
        return self._bytesize

    @bytesize.setter
    def bytesize(self, bytesize):
        if False:
            return 10
        'Change byte size.'
        if bytesize not in self.BYTESIZES:
            raise ValueError('Not a valid byte size: {!r}'.format(bytesize))
        self._bytesize = bytesize
        if self.is_open:
            self._reconfigure_port()

    @property
    def exclusive(self):
        if False:
            while True:
                i = 10
        'Get the current exclusive access setting.'
        return self._exclusive

    @exclusive.setter
    def exclusive(self, exclusive):
        if False:
            for i in range(10):
                print('nop')
        'Change the exclusive access setting.'
        self._exclusive = exclusive
        if self.is_open:
            self._reconfigure_port()

    @property
    def parity(self):
        if False:
            return 10
        'Get the current parity setting.'
        return self._parity

    @parity.setter
    def parity(self, parity):
        if False:
            for i in range(10):
                print('nop')
        'Change parity setting.'
        if parity not in self.PARITIES:
            raise ValueError('Not a valid parity: {!r}'.format(parity))
        self._parity = parity
        if self.is_open:
            self._reconfigure_port()

    @property
    def stopbits(self):
        if False:
            return 10
        'Get the current stop bits setting.'
        return self._stopbits

    @stopbits.setter
    def stopbits(self, stopbits):
        if False:
            while True:
                i = 10
        'Change stop bits size.'
        if stopbits not in self.STOPBITS:
            raise ValueError('Not a valid stop bit size: {!r}'.format(stopbits))
        self._stopbits = stopbits
        if self.is_open:
            self._reconfigure_port()

    @property
    def timeout(self):
        if False:
            return 10
        'Get the current timeout setting.'
        return self._timeout

    @timeout.setter
    def timeout(self, timeout):
        if False:
            return 10
        'Change timeout setting.'
        if timeout is not None:
            try:
                timeout + 1
            except TypeError:
                raise ValueError('Not a valid timeout: {!r}'.format(timeout))
            if timeout < 0:
                raise ValueError('Not a valid timeout: {!r}'.format(timeout))
        self._timeout = timeout
        if self.is_open:
            self._reconfigure_port()

    @property
    def write_timeout(self):
        if False:
            while True:
                i = 10
        'Get the current timeout setting.'
        return self._write_timeout

    @write_timeout.setter
    def write_timeout(self, timeout):
        if False:
            while True:
                i = 10
        'Change timeout setting.'
        if timeout is not None:
            if timeout < 0:
                raise ValueError('Not a valid timeout: {!r}'.format(timeout))
            try:
                timeout + 1
            except TypeError:
                raise ValueError('Not a valid timeout: {!r}'.format(timeout))
        self._write_timeout = timeout
        if self.is_open:
            self._reconfigure_port()

    @property
    def inter_byte_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the current inter-character timeout setting.'
        return self._inter_byte_timeout

    @inter_byte_timeout.setter
    def inter_byte_timeout(self, ic_timeout):
        if False:
            i = 10
            return i + 15
        'Change inter-byte timeout setting.'
        if ic_timeout is not None:
            if ic_timeout < 0:
                raise ValueError('Not a valid timeout: {!r}'.format(ic_timeout))
            try:
                ic_timeout + 1
            except TypeError:
                raise ValueError('Not a valid timeout: {!r}'.format(ic_timeout))
        self._inter_byte_timeout = ic_timeout
        if self.is_open:
            self._reconfigure_port()

    @property
    def xonxoff(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the current XON/XOFF setting.'
        return self._xonxoff

    @xonxoff.setter
    def xonxoff(self, xonxoff):
        if False:
            while True:
                i = 10
        'Change XON/XOFF setting.'
        self._xonxoff = xonxoff
        if self.is_open:
            self._reconfigure_port()

    @property
    def rtscts(self):
        if False:
            print('Hello World!')
        'Get the current RTS/CTS flow control setting.'
        return self._rtscts

    @rtscts.setter
    def rtscts(self, rtscts):
        if False:
            while True:
                i = 10
        'Change RTS/CTS flow control setting.'
        self._rtscts = rtscts
        if self.is_open:
            self._reconfigure_port()

    @property
    def dsrdtr(self):
        if False:
            while True:
                i = 10
        'Get the current DSR/DTR flow control setting.'
        return self._dsrdtr

    @dsrdtr.setter
    def dsrdtr(self, dsrdtr=None):
        if False:
            return 10
        'Change DsrDtr flow control setting.'
        if dsrdtr is None:
            self._dsrdtr = self._rtscts
        else:
            self._dsrdtr = dsrdtr
        if self.is_open:
            self._reconfigure_port()

    @property
    def rts(self):
        if False:
            print('Hello World!')
        return self._rts_state

    @rts.setter
    def rts(self, value):
        if False:
            return 10
        self._rts_state = value
        if self.is_open:
            self._update_rts_state()

    @property
    def dtr(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dtr_state

    @dtr.setter
    def dtr(self, value):
        if False:
            while True:
                i = 10
        self._dtr_state = value
        if self.is_open:
            self._update_dtr_state()

    @property
    def break_condition(self):
        if False:
            print('Hello World!')
        return self._break_state

    @break_condition.setter
    def break_condition(self, value):
        if False:
            i = 10
            return i + 15
        self._break_state = value
        if self.is_open:
            self._update_break_state()

    @property
    def rs485_mode(self):
        if False:
            print('Hello World!')
        '        Enable RS485 mode and apply new settings, set to None to disable.\n        See serial.rs485.RS485Settings for more info about the value.\n        '
        return self._rs485_mode

    @rs485_mode.setter
    def rs485_mode(self, rs485_settings):
        if False:
            for i in range(10):
                print('nop')
        self._rs485_mode = rs485_settings
        if self.is_open:
            self._reconfigure_port()
    _SAVED_SETTINGS = ('baudrate', 'bytesize', 'parity', 'stopbits', 'xonxoff', 'dsrdtr', 'rtscts', 'timeout', 'write_timeout', 'inter_byte_timeout')

    def get_settings(self):
        if False:
            for i in range(10):
                print('nop')
        '        Get current port settings as a dictionary. For use with\n        apply_settings().\n        '
        return dict([(key, getattr(self, '_' + key)) for key in self._SAVED_SETTINGS])

    def apply_settings(self, d):
        if False:
            while True:
                i = 10
        "        Apply stored settings from a dictionary returned from\n        get_settings(). It's allowed to delete keys from the dictionary. These\n        values will simply left unchanged.\n        "
        for key in self._SAVED_SETTINGS:
            if key in d and d[key] != getattr(self, '_' + key):
                setattr(self, key, d[key])

    def __repr__(self):
        if False:
            print('Hello World!')
        'String representation of the current port settings and its state.'
        return '{name}<id=0x{id:x}, open={p.is_open}>(port={p.portstr!r}, baudrate={p.baudrate!r}, bytesize={p.bytesize!r}, parity={p.parity!r}, stopbits={p.stopbits!r}, timeout={p.timeout!r}, xonxoff={p.xonxoff!r}, rtscts={p.rtscts!r}, dsrdtr={p.dsrdtr!r})'.format(name=self.__class__.__name__, id=id(self), p=self)

    def readable(self):
        if False:
            while True:
                i = 10
        return True

    def writable(self):
        if False:
            while True:
                i = 10
        return True

    def seekable(self):
        if False:
            while True:
                i = 10
        return False

    def readinto(self, b):
        if False:
            print('Hello World!')
        data = self.read(len(b))
        n = len(data)
        try:
            b[:n] = data
        except TypeError as err:
            import array
            if not isinstance(b, array.array):
                raise err
            b[:n] = array.array('b', data)
        return n

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    def closed(self):
        if False:
            while True:
                i = 10
        return not self.is_open

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        if self._port is not None and (not self.is_open):
            self.open()
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    def send_break(self, duration=0.25):
        if False:
            while True:
                i = 10
        '        Send break condition. Timed, returns to idle state after given\n        duration.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        self.break_condition = True
        time.sleep(duration)
        self.break_condition = False

    def flushInput(self):
        if False:
            return 10
        self.reset_input_buffer()

    def flushOutput(self):
        if False:
            while True:
                i = 10
        self.reset_output_buffer()

    def inWaiting(self):
        if False:
            return 10
        return self.in_waiting

    def sendBreak(self, duration=0.25):
        if False:
            while True:
                i = 10
        self.send_break(duration)

    def setRTS(self, value=1):
        if False:
            while True:
                i = 10
        self.rts = value

    def setDTR(self, value=1):
        if False:
            while True:
                i = 10
        self.dtr = value

    def getCTS(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cts

    def getDSR(self):
        if False:
            print('Hello World!')
        return self.dsr

    def getRI(self):
        if False:
            i = 10
            return i + 15
        return self.ri

    def getCD(self):
        if False:
            return 10
        return self.cd

    def setPort(self, port):
        if False:
            print('Hello World!')
        self.port = port

    @property
    def writeTimeout(self):
        if False:
            while True:
                i = 10
        return self.write_timeout

    @writeTimeout.setter
    def writeTimeout(self, timeout):
        if False:
            print('Hello World!')
        self.write_timeout = timeout

    @property
    def interCharTimeout(self):
        if False:
            return 10
        return self.inter_byte_timeout

    @interCharTimeout.setter
    def interCharTimeout(self, interCharTimeout):
        if False:
            return 10
        self.inter_byte_timeout = interCharTimeout

    def getSettingsDict(self):
        if False:
            while True:
                i = 10
        return self.get_settings()

    def applySettingsDict(self, d):
        if False:
            for i in range(10):
                print('nop')
        self.apply_settings(d)

    def isOpen(self):
        if False:
            return 10
        return self.is_open

    def read_all(self):
        if False:
            for i in range(10):
                print('nop')
        '        Read all bytes currently available in the buffer of the OS.\n        '
        return self.read(self.in_waiting)

    def read_until(self, expected=LF, size=None):
        if False:
            for i in range(10):
                print('nop')
        '        Read until an expected sequence is found (line feed by default), the size\n        is exceeded or until timeout occurs.\n        '
        lenterm = len(expected)
        line = bytearray()
        timeout = Timeout(self._timeout)
        while True:
            c = self.read(1)
            if c:
                line += c
                if line[-lenterm:] == expected:
                    break
                if size is not None and len(line) >= size:
                    break
            else:
                break
            if timeout.expired():
                break
        return bytes(line)

    def iread_until(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '        Read lines, implemented as generator. It will raise StopIteration on\n        timeout (empty read).\n        '
        while True:
            line = self.read_until(*args, **kwargs)
            if not line:
                break
            yield line
if __name__ == '__main__':
    import sys
    s = SerialBase()
    sys.stdout.write('port name:  {}\n'.format(s.name))
    sys.stdout.write('baud rates: {}\n'.format(s.BAUDRATES))
    sys.stdout.write('byte sizes: {}\n'.format(s.BYTESIZES))
    sys.stdout.write('parities:   {}\n'.format(s.PARITIES))
    sys.stdout.write('stop bits:  {}\n'.format(s.STOPBITS))
    sys.stdout.write('{}\n'.format(s))