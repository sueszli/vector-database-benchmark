from __future__ import absolute_import
import logging
import socket
import struct
import threading
import time
try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse
try:
    import Queue
except ImportError:
    import queue as Queue
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, iterbytes, PortNotOpenError, Timeout
LOGGER_LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
SE = b'\xf0'
NOP = b'\xf1'
DM = b'\xf2'
BRK = b'\xf3'
IP = b'\xf4'
AO = b'\xf5'
AYT = b'\xf6'
EC = b'\xf7'
EL = b'\xf8'
GA = b'\xf9'
SB = b'\xfa'
WILL = b'\xfb'
WONT = b'\xfc'
DO = b'\xfd'
DONT = b'\xfe'
IAC = b'\xff'
IAC_DOUBLED = b'\xff\xff'
BINARY = b'\x00'
ECHO = b'\x01'
SGA = b'\x03'
COM_PORT_OPTION = b','
SET_BAUDRATE = b'\x01'
SET_DATASIZE = b'\x02'
SET_PARITY = b'\x03'
SET_STOPSIZE = b'\x04'
SET_CONTROL = b'\x05'
NOTIFY_LINESTATE = b'\x06'
NOTIFY_MODEMSTATE = b'\x07'
FLOWCONTROL_SUSPEND = b'\x08'
FLOWCONTROL_RESUME = b'\t'
SET_LINESTATE_MASK = b'\n'
SET_MODEMSTATE_MASK = b'\x0b'
PURGE_DATA = b'\x0c'
SERVER_SET_BAUDRATE = b'e'
SERVER_SET_DATASIZE = b'f'
SERVER_SET_PARITY = b'g'
SERVER_SET_STOPSIZE = b'h'
SERVER_SET_CONTROL = b'i'
SERVER_NOTIFY_LINESTATE = b'j'
SERVER_NOTIFY_MODEMSTATE = b'k'
SERVER_FLOWCONTROL_SUSPEND = b'l'
SERVER_FLOWCONTROL_RESUME = b'm'
SERVER_SET_LINESTATE_MASK = b'n'
SERVER_SET_MODEMSTATE_MASK = b'o'
SERVER_PURGE_DATA = b'p'
RFC2217_ANSWER_MAP = {SET_BAUDRATE: SERVER_SET_BAUDRATE, SET_DATASIZE: SERVER_SET_DATASIZE, SET_PARITY: SERVER_SET_PARITY, SET_STOPSIZE: SERVER_SET_STOPSIZE, SET_CONTROL: SERVER_SET_CONTROL, NOTIFY_LINESTATE: SERVER_NOTIFY_LINESTATE, NOTIFY_MODEMSTATE: SERVER_NOTIFY_MODEMSTATE, FLOWCONTROL_SUSPEND: SERVER_FLOWCONTROL_SUSPEND, FLOWCONTROL_RESUME: SERVER_FLOWCONTROL_RESUME, SET_LINESTATE_MASK: SERVER_SET_LINESTATE_MASK, SET_MODEMSTATE_MASK: SERVER_SET_MODEMSTATE_MASK, PURGE_DATA: SERVER_PURGE_DATA}
SET_CONTROL_REQ_FLOW_SETTING = b'\x00'
SET_CONTROL_USE_NO_FLOW_CONTROL = b'\x01'
SET_CONTROL_USE_SW_FLOW_CONTROL = b'\x02'
SET_CONTROL_USE_HW_FLOW_CONTROL = b'\x03'
SET_CONTROL_REQ_BREAK_STATE = b'\x04'
SET_CONTROL_BREAK_ON = b'\x05'
SET_CONTROL_BREAK_OFF = b'\x06'
SET_CONTROL_REQ_DTR = b'\x07'
SET_CONTROL_DTR_ON = b'\x08'
SET_CONTROL_DTR_OFF = b'\t'
SET_CONTROL_REQ_RTS = b'\n'
SET_CONTROL_RTS_ON = b'\x0b'
SET_CONTROL_RTS_OFF = b'\x0c'
SET_CONTROL_REQ_FLOW_SETTING_IN = b'\r'
SET_CONTROL_USE_NO_FLOW_CONTROL_IN = b'\x0e'
SET_CONTROL_USE_SW_FLOW_CONTOL_IN = b'\x0f'
SET_CONTROL_USE_HW_FLOW_CONTOL_IN = b'\x10'
SET_CONTROL_USE_DCD_FLOW_CONTROL = b'\x11'
SET_CONTROL_USE_DTR_FLOW_CONTROL = b'\x12'
SET_CONTROL_USE_DSR_FLOW_CONTROL = b'\x13'
LINESTATE_MASK_TIMEOUT = 128
LINESTATE_MASK_SHIFTREG_EMPTY = 64
LINESTATE_MASK_TRANSREG_EMPTY = 32
LINESTATE_MASK_BREAK_DETECT = 16
LINESTATE_MASK_FRAMING_ERROR = 8
LINESTATE_MASK_PARTIY_ERROR = 4
LINESTATE_MASK_OVERRUN_ERROR = 2
LINESTATE_MASK_DATA_READY = 1
MODEMSTATE_MASK_CD = 128
MODEMSTATE_MASK_RI = 64
MODEMSTATE_MASK_DSR = 32
MODEMSTATE_MASK_CTS = 16
MODEMSTATE_MASK_CD_CHANGE = 8
MODEMSTATE_MASK_RI_CHANGE = 4
MODEMSTATE_MASK_DSR_CHANGE = 2
MODEMSTATE_MASK_CTS_CHANGE = 1
PURGE_RECEIVE_BUFFER = b'\x01'
PURGE_TRANSMIT_BUFFER = b'\x02'
PURGE_BOTH_BUFFERS = b'\x03'
RFC2217_PARITY_MAP = {serial.PARITY_NONE: 1, serial.PARITY_ODD: 2, serial.PARITY_EVEN: 3, serial.PARITY_MARK: 4, serial.PARITY_SPACE: 5}
RFC2217_REVERSE_PARITY_MAP = dict(((v, k) for (k, v) in RFC2217_PARITY_MAP.items()))
RFC2217_STOPBIT_MAP = {serial.STOPBITS_ONE: 1, serial.STOPBITS_ONE_POINT_FIVE: 3, serial.STOPBITS_TWO: 2}
RFC2217_REVERSE_STOPBIT_MAP = dict(((v, k) for (k, v) in RFC2217_STOPBIT_MAP.items()))
M_NORMAL = 0
M_IAC_SEEN = 1
M_NEGOTIATE = 2
REQUESTED = 'REQUESTED'
ACTIVE = 'ACTIVE'
INACTIVE = 'INACTIVE'
REALLY_INACTIVE = 'REALLY_INACTIVE'

class TelnetOption(object):
    """Manage a single telnet option, keeps track of DO/DONT WILL/WONT."""

    def __init__(self, connection, name, option, send_yes, send_no, ack_yes, ack_no, initial_state, activation_callback=None):
        if False:
            return 10
        '        Initialize option.\n        :param connection: connection used to transmit answers\n        :param name: a readable name for debug outputs\n        :param send_yes: what to send when option is to be enabled.\n        :param send_no: what to send when option is to be disabled.\n        :param ack_yes: what to expect when remote agrees on option.\n        :param ack_no: what to expect when remote disagrees on option.\n        :param initial_state: options initialized with REQUESTED are tried to\n            be enabled on startup. use INACTIVE for all others.\n        '
        self.connection = connection
        self.name = name
        self.option = option
        self.send_yes = send_yes
        self.send_no = send_no
        self.ack_yes = ack_yes
        self.ack_no = ack_no
        self.state = initial_state
        self.active = False
        self.activation_callback = activation_callback

    def __repr__(self):
        if False:
            while True:
                i = 10
        'String for debug outputs'
        return '{o.name}:{o.active}({o.state})'.format(o=self)

    def process_incoming(self, command):
        if False:
            for i in range(10):
                print('nop')
        '        A DO/DONT/WILL/WONT was received for this option, update state and\n        answer when needed.\n        '
        if command == self.ack_yes:
            if self.state is REQUESTED:
                self.state = ACTIVE
                self.active = True
                if self.activation_callback is not None:
                    self.activation_callback()
            elif self.state is ACTIVE:
                pass
            elif self.state is INACTIVE:
                self.state = ACTIVE
                self.connection.telnet_send_option(self.send_yes, self.option)
                self.active = True
                if self.activation_callback is not None:
                    self.activation_callback()
            elif self.state is REALLY_INACTIVE:
                self.connection.telnet_send_option(self.send_no, self.option)
            else:
                raise ValueError('option in illegal state {!r}'.format(self))
        elif command == self.ack_no:
            if self.state is REQUESTED:
                self.state = INACTIVE
                self.active = False
            elif self.state is ACTIVE:
                self.state = INACTIVE
                self.connection.telnet_send_option(self.send_no, self.option)
                self.active = False
            elif self.state is INACTIVE:
                pass
            elif self.state is REALLY_INACTIVE:
                pass
            else:
                raise ValueError('option in illegal state {!r}'.format(self))

class TelnetSubnegotiation(object):
    """    A object to handle subnegotiation of options. In this case actually
    sub-sub options for RFC 2217. It is used to track com port options.
    """

    def __init__(self, connection, name, option, ack_option=None):
        if False:
            print('Hello World!')
        if ack_option is None:
            ack_option = option
        self.connection = connection
        self.name = name
        self.option = option
        self.value = None
        self.ack_option = ack_option
        self.state = INACTIVE

    def __repr__(self):
        if False:
            print('Hello World!')
        'String for debug outputs.'
        return '{sn.name}:{sn.state}'.format(sn=self)

    def set(self, value):
        if False:
            while True:
                i = 10
        '        Request a change of the value. a request is sent to the server. if\n        the client needs to know if the change is performed he has to check the\n        state of this object.\n        '
        self.value = value
        self.state = REQUESTED
        self.connection.rfc2217_send_subnegotiation(self.option, self.value)
        if self.connection.logger:
            self.connection.logger.debug('SB Requesting {} -> {!r}'.format(self.name, self.value))

    def is_ready(self):
        if False:
            print('Hello World!')
        '        Check if answer from server has been received. when server rejects\n        the change, raise a ValueError.\n        '
        if self.state == REALLY_INACTIVE:
            raise ValueError('remote rejected value for option {!r}'.format(self.name))
        return self.state == ACTIVE
    active = property(is_ready)

    def wait(self, timeout=3):
        if False:
            for i in range(10):
                print('nop')
        '        Wait until the subnegotiation has been acknowledged or timeout. It\n        can also throw a value error when the answer from the server does not\n        match the value sent.\n        '
        timeout_timer = Timeout(timeout)
        while not timeout_timer.expired():
            time.sleep(0.05)
            if self.is_ready():
                break
        else:
            raise SerialException('timeout while waiting for option {!r}'.format(self.name))

    def check_answer(self, suboption):
        if False:
            return 10
        '        Check an incoming subnegotiation block. The parameter already has\n        cut off the header like sub option number and com port option value.\n        '
        if self.value == suboption[:len(self.value)]:
            self.state = ACTIVE
        else:
            self.state = REALLY_INACTIVE
        if self.connection.logger:
            self.connection.logger.debug('SB Answer {} -> {!r} -> {}'.format(self.name, suboption, self.state))

class Serial(SerialBase):
    """Serial port implementation for RFC 2217 remote serial ports."""
    BAUDRATES = (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200)

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._thread = None
        self._socket = None
        self._linestate = 0
        self._modemstate = None
        self._modemstate_timeout = Timeout(-1)
        self._remote_suspend_flow = False
        self._write_lock = None
        self.logger = None
        self._ignore_set_control_answer = False
        self._poll_modem_state = False
        self._network_timeout = 3
        self._telnet_options = None
        self._rfc2217_port_settings = None
        self._rfc2217_options = None
        self._read_buffer = None
        super(Serial, self).__init__(*args, **kwargs)

    def open(self):
        if False:
            while True:
                i = 10
        '        Open port with current settings. This may throw a SerialException\n        if the port cannot be opened.\n        '
        self.logger = None
        self._ignore_set_control_answer = False
        self._poll_modem_state = False
        self._network_timeout = 3
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if self.is_open:
            raise SerialException('Port is already open.')
        try:
            self._socket = socket.create_connection(self.from_url(self.portstr), timeout=5)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as msg:
            self._socket = None
            raise SerialException('Could not open port {}: {}'.format(self.portstr, msg))
        self._read_buffer = Queue.Queue()
        self._write_lock = threading.Lock()
        mandadory_options = [TelnetOption(self, 'we-BINARY', BINARY, WILL, WONT, DO, DONT, INACTIVE), TelnetOption(self, 'we-RFC2217', COM_PORT_OPTION, WILL, WONT, DO, DONT, REQUESTED)]
        self._telnet_options = [TelnetOption(self, 'ECHO', ECHO, DO, DONT, WILL, WONT, REQUESTED), TelnetOption(self, 'we-SGA', SGA, WILL, WONT, DO, DONT, REQUESTED), TelnetOption(self, 'they-SGA', SGA, DO, DONT, WILL, WONT, REQUESTED), TelnetOption(self, 'they-BINARY', BINARY, DO, DONT, WILL, WONT, INACTIVE), TelnetOption(self, 'they-RFC2217', COM_PORT_OPTION, DO, DONT, WILL, WONT, REQUESTED)] + mandadory_options
        self._rfc2217_port_settings = {'baudrate': TelnetSubnegotiation(self, 'baudrate', SET_BAUDRATE, SERVER_SET_BAUDRATE), 'datasize': TelnetSubnegotiation(self, 'datasize', SET_DATASIZE, SERVER_SET_DATASIZE), 'parity': TelnetSubnegotiation(self, 'parity', SET_PARITY, SERVER_SET_PARITY), 'stopsize': TelnetSubnegotiation(self, 'stopsize', SET_STOPSIZE, SERVER_SET_STOPSIZE)}
        self._rfc2217_options = {'purge': TelnetSubnegotiation(self, 'purge', PURGE_DATA, SERVER_PURGE_DATA), 'control': TelnetSubnegotiation(self, 'control', SET_CONTROL, SERVER_SET_CONTROL)}
        self._rfc2217_options.update(self._rfc2217_port_settings)
        self._linestate = 0
        self._modemstate = None
        self._modemstate_timeout = Timeout(-1)
        self._remote_suspend_flow = False
        self.is_open = True
        self._thread = threading.Thread(target=self._telnet_read_loop)
        self._thread.daemon = True
        self._thread.setName('pySerial RFC 2217 reader thread for {}'.format(self._port))
        self._thread.start()
        try:
            for option in self._telnet_options:
                if option.state is REQUESTED:
                    self.telnet_send_option(option.send_yes, option.option)
            timeout = Timeout(self._network_timeout)
            while not timeout.expired():
                time.sleep(0.05)
                if sum((o.active for o in mandadory_options)) == sum((o.state != INACTIVE for o in mandadory_options)):
                    break
            else:
                raise SerialException('Remote does not seem to support RFC2217 or BINARY mode {!r}'.format(mandadory_options))
            if self.logger:
                self.logger.info('Negotiated options: {}'.format(self._telnet_options))
            self._reconfigure_port()
            if not self._dsrdtr:
                self._update_dtr_state()
            if not self._rtscts:
                self._update_rts_state()
            self.reset_input_buffer()
            self.reset_output_buffer()
        except:
            self.close()
            raise

    def _reconfigure_port(self):
        if False:
            while True:
                i = 10
        'Set communication parameters on opened port.'
        if self._socket is None:
            raise SerialException('Can only operate on open ports')
        if self._write_timeout is not None:
            raise NotImplementedError('write_timeout is currently not supported')
        if not 0 < self._baudrate < 2 ** 32:
            raise ValueError('invalid baudrate: {!r}'.format(self._baudrate))
        self._rfc2217_port_settings['baudrate'].set(struct.pack(b'!I', self._baudrate))
        self._rfc2217_port_settings['datasize'].set(struct.pack(b'!B', self._bytesize))
        self._rfc2217_port_settings['parity'].set(struct.pack(b'!B', RFC2217_PARITY_MAP[self._parity]))
        self._rfc2217_port_settings['stopsize'].set(struct.pack(b'!B', RFC2217_STOPBIT_MAP[self._stopbits]))
        items = self._rfc2217_port_settings.values()
        if self.logger:
            self.logger.debug('Negotiating settings: {}'.format(items))
        timeout = Timeout(self._network_timeout)
        while not timeout.expired():
            time.sleep(0.05)
            if sum((o.active for o in items)) == len(items):
                break
        else:
            raise SerialException('Remote does not accept parameter change (RFC2217): {!r}'.format(items))
        if self.logger:
            self.logger.info('Negotiated settings: {}'.format(items))
        if self._rtscts and self._xonxoff:
            raise ValueError('xonxoff and rtscts together are not supported')
        elif self._rtscts:
            self.rfc2217_set_control(SET_CONTROL_USE_HW_FLOW_CONTROL)
        elif self._xonxoff:
            self.rfc2217_set_control(SET_CONTROL_USE_SW_FLOW_CONTROL)
        else:
            self.rfc2217_set_control(SET_CONTROL_USE_NO_FLOW_CONTROL)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        'Close port'
        self.is_open = False
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
            except:
                pass
        if self._thread:
            self._thread.join(7)
            self._thread = None
            time.sleep(0.3)
        self._socket = None

    def from_url(self, url):
        if False:
            for i in range(10):
                print('nop')
        '        extract host and port from an URL string, other settings are extracted\n        an stored in instance\n        '
        parts = urlparse.urlsplit(url)
        if parts.scheme != 'rfc2217':
            raise SerialException('expected a string in the form "rfc2217://<host>:<port>[?option[&option...]]": not starting with rfc2217:// ({!r})'.format(parts.scheme))
        try:
            for (option, values) in urlparse.parse_qs(parts.query, True).items():
                if option == 'logging':
                    logging.basicConfig()
                    self.logger = logging.getLogger('pySerial.rfc2217')
                    self.logger.setLevel(LOGGER_LEVELS[values[0]])
                    self.logger.debug('enabled logging')
                elif option == 'ign_set_control':
                    self._ignore_set_control_answer = True
                elif option == 'poll_modem':
                    self._poll_modem_state = True
                elif option == 'timeout':
                    self._network_timeout = float(values[0])
                else:
                    raise ValueError('unknown option: {!r}'.format(option))
            if not 0 <= parts.port < 65536:
                raise ValueError('port not in range 0...65535')
        except ValueError as e:
            raise SerialException('expected a string in the form "rfc2217://<host>:<port>[?option[&option...]]": {}'.format(e))
        return (parts.hostname, parts.port)

    @property
    def in_waiting(self):
        if False:
            i = 10
            return i + 15
        'Return the number of bytes currently in the input buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        return self._read_buffer.qsize()

    def read(self, size=1):
        if False:
            for i in range(10):
                print('nop')
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        data = bytearray()
        try:
            timeout = Timeout(self._timeout)
            while len(data) < size:
                if self._thread is None or not self._thread.is_alive():
                    raise SerialException('connection failed (reader thread died)')
                buf = self._read_buffer.get(True, timeout.time_left())
                if buf is None:
                    return bytes(data)
                data += buf
                if timeout.expired():
                    break
        except Queue.Empty:
            pass
        return bytes(data)

    def write(self, data):
        if False:
            return 10
        '        Output the given byte string over the serial port. Can block if the\n        connection is blocked. May raise SerialException if the connection is\n        closed.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        with self._write_lock:
            try:
                self._socket.sendall(to_bytes(data).replace(IAC, IAC_DOUBLED))
            except socket.error as e:
                raise SerialException('connection failed (socket error): {}'.format(e))
        return len(data)

    def reset_input_buffer(self):
        if False:
            print('Hello World!')
        'Clear input buffer, discarding all that is in the buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        self.rfc2217_send_purge(PURGE_RECEIVE_BUFFER)
        while self._read_buffer.qsize():
            self._read_buffer.get(False)

    def reset_output_buffer(self):
        if False:
            print('Hello World!')
        '        Clear output buffer, aborting the current output and\n        discarding all that is in the buffer.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        self.rfc2217_send_purge(PURGE_TRANSMIT_BUFFER)

    def _update_break_state(self):
        if False:
            print('Hello World!')
        '        Set break: Controls TXD. When active, to transmitting is\n        possible.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('set BREAK to {}'.format('active' if self._break_state else 'inactive'))
        if self._break_state:
            self.rfc2217_set_control(SET_CONTROL_BREAK_ON)
        else:
            self.rfc2217_set_control(SET_CONTROL_BREAK_OFF)

    def _update_rts_state(self):
        if False:
            return 10
        'Set terminal status line: Request To Send.'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('set RTS to {}'.format('active' if self._rts_state else 'inactive'))
        if self._rts_state:
            self.rfc2217_set_control(SET_CONTROL_RTS_ON)
        else:
            self.rfc2217_set_control(SET_CONTROL_RTS_OFF)

    def _update_dtr_state(self):
        if False:
            while True:
                i = 10
        'Set terminal status line: Data Terminal Ready.'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('set DTR to {}'.format('active' if self._dtr_state else 'inactive'))
        if self._dtr_state:
            self.rfc2217_set_control(SET_CONTROL_DTR_ON)
        else:
            self.rfc2217_set_control(SET_CONTROL_DTR_OFF)

    @property
    def cts(self):
        if False:
            print('Hello World!')
        'Read terminal status line: Clear To Send.'
        if not self.is_open:
            raise PortNotOpenError()
        return bool(self.get_modem_state() & MODEMSTATE_MASK_CTS)

    @property
    def dsr(self):
        if False:
            while True:
                i = 10
        'Read terminal status line: Data Set Ready.'
        if not self.is_open:
            raise PortNotOpenError()
        return bool(self.get_modem_state() & MODEMSTATE_MASK_DSR)

    @property
    def ri(self):
        if False:
            return 10
        'Read terminal status line: Ring Indicator.'
        if not self.is_open:
            raise PortNotOpenError()
        return bool(self.get_modem_state() & MODEMSTATE_MASK_RI)

    @property
    def cd(self):
        if False:
            for i in range(10):
                print('nop')
        'Read terminal status line: Carrier Detect.'
        if not self.is_open:
            raise PortNotOpenError()
        return bool(self.get_modem_state() & MODEMSTATE_MASK_CD)

    def _telnet_read_loop(self):
        if False:
            print('Hello World!')
        'Read loop for the socket.'
        mode = M_NORMAL
        suboption = None
        try:
            while self.is_open:
                try:
                    data = self._socket.recv(1024)
                except socket.timeout:
                    continue
                except socket.error as e:
                    if self.logger:
                        self.logger.debug('socket error in reader thread: {}'.format(e))
                    self._read_buffer.put(None)
                    break
                if not data:
                    self._read_buffer.put(None)
                    break
                for byte in iterbytes(data):
                    if mode == M_NORMAL:
                        if byte == IAC:
                            mode = M_IAC_SEEN
                        elif suboption is not None:
                            suboption += byte
                        else:
                            self._read_buffer.put(byte)
                    elif mode == M_IAC_SEEN:
                        if byte == IAC:
                            if suboption is not None:
                                suboption += IAC
                            else:
                                self._read_buffer.put(IAC)
                            mode = M_NORMAL
                        elif byte == SB:
                            suboption = bytearray()
                            mode = M_NORMAL
                        elif byte == SE:
                            self._telnet_process_subnegotiation(bytes(suboption))
                            suboption = None
                            mode = M_NORMAL
                        elif byte in (DO, DONT, WILL, WONT):
                            telnet_command = byte
                            mode = M_NEGOTIATE
                        else:
                            self._telnet_process_command(byte)
                            mode = M_NORMAL
                    elif mode == M_NEGOTIATE:
                        self._telnet_negotiate_option(telnet_command, byte)
                        mode = M_NORMAL
        finally:
            if self.logger:
                self.logger.debug('read thread terminated')

    def _telnet_process_command(self, command):
        if False:
            print('Hello World!')
        'Process commands other than DO, DONT, WILL, WONT.'
        if self.logger:
            self.logger.warning('ignoring Telnet command: {!r}'.format(command))

    def _telnet_negotiate_option(self, command, option):
        if False:
            print('Hello World!')
        'Process incoming DO, DONT, WILL, WONT.'
        known = False
        for item in self._telnet_options:
            if item.option == option:
                item.process_incoming(command)
                known = True
        if not known:
            if command == WILL or command == DO:
                self.telnet_send_option(DONT if command == WILL else WONT, option)
                if self.logger:
                    self.logger.warning('rejected Telnet option: {!r}'.format(option))

    def _telnet_process_subnegotiation(self, suboption):
        if False:
            while True:
                i = 10
        'Process subnegotiation, the data between IAC SB and IAC SE.'
        if suboption[0:1] == COM_PORT_OPTION:
            if suboption[1:2] == SERVER_NOTIFY_LINESTATE and len(suboption) >= 3:
                self._linestate = ord(suboption[2:3])
                if self.logger:
                    self.logger.info('NOTIFY_LINESTATE: {}'.format(self._linestate))
            elif suboption[1:2] == SERVER_NOTIFY_MODEMSTATE and len(suboption) >= 3:
                self._modemstate = ord(suboption[2:3])
                if self.logger:
                    self.logger.info('NOTIFY_MODEMSTATE: {}'.format(self._modemstate))
                self._modemstate_timeout.restart(0.3)
            elif suboption[1:2] == FLOWCONTROL_SUSPEND:
                self._remote_suspend_flow = True
            elif suboption[1:2] == FLOWCONTROL_RESUME:
                self._remote_suspend_flow = False
            else:
                for item in self._rfc2217_options.values():
                    if item.ack_option == suboption[1:2]:
                        item.check_answer(bytes(suboption[2:]))
                        break
                else:
                    if self.logger:
                        self.logger.warning('ignoring COM_PORT_OPTION: {!r}'.format(suboption))
        elif self.logger:
            self.logger.warning('ignoring subnegotiation: {!r}'.format(suboption))

    def _internal_raw_write(self, data):
        if False:
            return 10
        'internal socket write with no data escaping. used to send telnet stuff.'
        with self._write_lock:
            self._socket.sendall(data)

    def telnet_send_option(self, action, option):
        if False:
            print('Hello World!')
        'Send DO, DONT, WILL, WONT.'
        self._internal_raw_write(IAC + action + option)

    def rfc2217_send_subnegotiation(self, option, value=b''):
        if False:
            i = 10
            return i + 15
        'Subnegotiation of RFC2217 parameters.'
        value = value.replace(IAC, IAC_DOUBLED)
        self._internal_raw_write(IAC + SB + COM_PORT_OPTION + option + value + IAC + SE)

    def rfc2217_send_purge(self, value):
        if False:
            return 10
        '        Send purge request to the remote.\n        (PURGE_RECEIVE_BUFFER / PURGE_TRANSMIT_BUFFER / PURGE_BOTH_BUFFERS)\n        '
        item = self._rfc2217_options['purge']
        item.set(value)
        item.wait(self._network_timeout)

    def rfc2217_set_control(self, value):
        if False:
            i = 10
            return i + 15
        'transmit change of control line to remote'
        item = self._rfc2217_options['control']
        item.set(value)
        if self._ignore_set_control_answer:
            time.sleep(0.1)
        else:
            item.wait(self._network_timeout)

    def rfc2217_flow_server_ready(self):
        if False:
            i = 10
            return i + 15
        '        check if server is ready to receive data. block for some time when\n        not.\n        '

    def get_modem_state(self):
        if False:
            while True:
                i = 10
        '        get last modem state (cached value. If value is "old", request a new\n        one. This cache helps that we don\'t issue to many requests when e.g. all\n        status lines, one after the other is queried by the user (CTS, DSR\n        etc.)\n        '
        if self._poll_modem_state and self._modemstate_timeout.expired():
            if self.logger:
                self.logger.debug('polling modem state')
            self.rfc2217_send_subnegotiation(NOTIFY_MODEMSTATE)
            timeout = Timeout(self._network_timeout)
            while not timeout.expired():
                time.sleep(0.05)
                if not self._modemstate_timeout.expired():
                    break
            else:
                if self.logger:
                    self.logger.warning('poll for modem state failed')
        if self._modemstate is not None:
            if self.logger:
                self.logger.debug('using cached modem state')
            return self._modemstate
        else:
            raise SerialException('remote sends no NOTIFY_MODEMSTATE')

class PortManager(object):
    """    This class manages the state of Telnet and RFC 2217. It needs a serial
    instance and a connection to work with. Connection is expected to implement
    a (thread safe) write function, that writes the string to the network.
    """

    def __init__(self, serial_port, connection, logger=None):
        if False:
            for i in range(10):
                print('nop')
        self.serial = serial_port
        self.connection = connection
        self.logger = logger
        self._client_is_rfc2217 = False
        self.mode = M_NORMAL
        self.suboption = None
        self.telnet_command = None
        self.modemstate_mask = 255
        self.last_modemstate = None
        self.linstate_mask = 0
        self._telnet_options = [TelnetOption(self, 'ECHO', ECHO, WILL, WONT, DO, DONT, REQUESTED), TelnetOption(self, 'we-SGA', SGA, WILL, WONT, DO, DONT, REQUESTED), TelnetOption(self, 'they-SGA', SGA, DO, DONT, WILL, WONT, INACTIVE), TelnetOption(self, 'we-BINARY', BINARY, WILL, WONT, DO, DONT, INACTIVE), TelnetOption(self, 'they-BINARY', BINARY, DO, DONT, WILL, WONT, REQUESTED), TelnetOption(self, 'we-RFC2217', COM_PORT_OPTION, WILL, WONT, DO, DONT, REQUESTED, self._client_ok), TelnetOption(self, 'they-RFC2217', COM_PORT_OPTION, DO, DONT, WILL, WONT, INACTIVE, self._client_ok)]
        if self.logger:
            self.logger.debug('requesting initial Telnet/RFC 2217 options')
        for option in self._telnet_options:
            if option.state is REQUESTED:
                self.telnet_send_option(option.send_yes, option.option)

    def _client_ok(self):
        if False:
            return 10
        '        callback of telnet option. It gets called when option is activated.\n        This one here is used to detect when the client agrees on RFC 2217. A\n        flag is set so that other functions like check_modem_lines know if the\n        client is OK.\n        '
        self._client_is_rfc2217 = True
        if self.logger:
            self.logger.info('client accepts RFC 2217')
        self.check_modem_lines(force_notification=True)

    def telnet_send_option(self, action, option):
        if False:
            print('Hello World!')
        'Send DO, DONT, WILL, WONT.'
        self.connection.write(IAC + action + option)

    def rfc2217_send_subnegotiation(self, option, value=b''):
        if False:
            print('Hello World!')
        'Subnegotiation of RFC 2217 parameters.'
        value = value.replace(IAC, IAC_DOUBLED)
        self.connection.write(IAC + SB + COM_PORT_OPTION + option + value + IAC + SE)

    def check_modem_lines(self, force_notification=False):
        if False:
            return 10
        '        read control lines from serial port and compare the last value sent to remote.\n        send updates on changes.\n        '
        modemstate = (self.serial.cts and MODEMSTATE_MASK_CTS) | (self.serial.dsr and MODEMSTATE_MASK_DSR) | (self.serial.ri and MODEMSTATE_MASK_RI) | (self.serial.cd and MODEMSTATE_MASK_CD)
        deltas = modemstate ^ (self.last_modemstate or 0)
        if deltas & MODEMSTATE_MASK_CTS:
            modemstate |= MODEMSTATE_MASK_CTS_CHANGE
        if deltas & MODEMSTATE_MASK_DSR:
            modemstate |= MODEMSTATE_MASK_DSR_CHANGE
        if deltas & MODEMSTATE_MASK_RI:
            modemstate |= MODEMSTATE_MASK_RI_CHANGE
        if deltas & MODEMSTATE_MASK_CD:
            modemstate |= MODEMSTATE_MASK_CD_CHANGE
        if modemstate != self.last_modemstate or force_notification:
            if self._client_is_rfc2217 and modemstate & self.modemstate_mask or force_notification:
                self.rfc2217_send_subnegotiation(SERVER_NOTIFY_MODEMSTATE, to_bytes([modemstate & self.modemstate_mask]))
                if self.logger:
                    self.logger.info('NOTIFY_MODEMSTATE: {}'.format(modemstate))
            self.last_modemstate = modemstate & 240

    def escape(self, data):
        if False:
            return 10
        '        This generator function is for the user. All outgoing data has to be\n        properly escaped, so that no IAC character in the data stream messes up\n        the Telnet state machine in the server.\n\n        socket.sendall(escape(data))\n        '
        for byte in iterbytes(data):
            if byte == IAC:
                yield IAC
                yield IAC
            else:
                yield byte

    def filter(self, data):
        if False:
            return 10
        '        Handle a bunch of incoming bytes. This is a generator. It will yield\n        all characters not of interest for Telnet/RFC 2217.\n\n        The idea is that the reader thread pushes data from the socket through\n        this filter:\n\n        for byte in filter(socket.recv(1024)):\n            # do things like CR/LF conversion/whatever\n            # and write data to the serial port\n            serial.write(byte)\n\n        (socket error handling code left as exercise for the reader)\n        '
        for byte in iterbytes(data):
            if self.mode == M_NORMAL:
                if byte == IAC:
                    self.mode = M_IAC_SEEN
                elif self.suboption is not None:
                    self.suboption += byte
                else:
                    yield byte
            elif self.mode == M_IAC_SEEN:
                if byte == IAC:
                    if self.suboption is not None:
                        self.suboption += byte
                    else:
                        yield byte
                    self.mode = M_NORMAL
                elif byte == SB:
                    self.suboption = bytearray()
                    self.mode = M_NORMAL
                elif byte == SE:
                    self._telnet_process_subnegotiation(bytes(self.suboption))
                    self.suboption = None
                    self.mode = M_NORMAL
                elif byte in (DO, DONT, WILL, WONT):
                    self.telnet_command = byte
                    self.mode = M_NEGOTIATE
                else:
                    self._telnet_process_command(byte)
                    self.mode = M_NORMAL
            elif self.mode == M_NEGOTIATE:
                self._telnet_negotiate_option(self.telnet_command, byte)
                self.mode = M_NORMAL

    def _telnet_process_command(self, command):
        if False:
            return 10
        'Process commands other than DO, DONT, WILL, WONT.'
        if self.logger:
            self.logger.warning('ignoring Telnet command: {!r}'.format(command))

    def _telnet_negotiate_option(self, command, option):
        if False:
            for i in range(10):
                print('nop')
        'Process incoming DO, DONT, WILL, WONT.'
        known = False
        for item in self._telnet_options:
            if item.option == option:
                item.process_incoming(command)
                known = True
        if not known:
            if command == WILL or command == DO:
                self.telnet_send_option(DONT if command == WILL else WONT, option)
                if self.logger:
                    self.logger.warning('rejected Telnet option: {!r}'.format(option))

    def _telnet_process_subnegotiation(self, suboption):
        if False:
            for i in range(10):
                print('nop')
        'Process subnegotiation, the data between IAC SB and IAC SE.'
        if suboption[0:1] == COM_PORT_OPTION:
            if self.logger:
                self.logger.debug('received COM_PORT_OPTION: {!r}'.format(suboption))
            if suboption[1:2] == SET_BAUDRATE:
                backup = self.serial.baudrate
                try:
                    (baudrate,) = struct.unpack(b'!I', suboption[2:6])
                    if baudrate != 0:
                        self.serial.baudrate = baudrate
                except ValueError as e:
                    if self.logger:
                        self.logger.error('failed to set baud rate: {}'.format(e))
                    self.serial.baudrate = backup
                else:
                    if self.logger:
                        self.logger.info('{} baud rate: {}'.format('set' if baudrate else 'get', self.serial.baudrate))
                self.rfc2217_send_subnegotiation(SERVER_SET_BAUDRATE, struct.pack(b'!I', self.serial.baudrate))
            elif suboption[1:2] == SET_DATASIZE:
                backup = self.serial.bytesize
                try:
                    (datasize,) = struct.unpack(b'!B', suboption[2:3])
                    if datasize != 0:
                        self.serial.bytesize = datasize
                except ValueError as e:
                    if self.logger:
                        self.logger.error('failed to set data size: {}'.format(e))
                    self.serial.bytesize = backup
                else:
                    if self.logger:
                        self.logger.info('{} data size: {}'.format('set' if datasize else 'get', self.serial.bytesize))
                self.rfc2217_send_subnegotiation(SERVER_SET_DATASIZE, struct.pack(b'!B', self.serial.bytesize))
            elif suboption[1:2] == SET_PARITY:
                backup = self.serial.parity
                try:
                    parity = struct.unpack(b'!B', suboption[2:3])[0]
                    if parity != 0:
                        self.serial.parity = RFC2217_REVERSE_PARITY_MAP[parity]
                except ValueError as e:
                    if self.logger:
                        self.logger.error('failed to set parity: {}'.format(e))
                    self.serial.parity = backup
                else:
                    if self.logger:
                        self.logger.info('{} parity: {}'.format('set' if parity else 'get', self.serial.parity))
                self.rfc2217_send_subnegotiation(SERVER_SET_PARITY, struct.pack(b'!B', RFC2217_PARITY_MAP[self.serial.parity]))
            elif suboption[1:2] == SET_STOPSIZE:
                backup = self.serial.stopbits
                try:
                    stopbits = struct.unpack(b'!B', suboption[2:3])[0]
                    if stopbits != 0:
                        self.serial.stopbits = RFC2217_REVERSE_STOPBIT_MAP[stopbits]
                except ValueError as e:
                    if self.logger:
                        self.logger.error('failed to set stop bits: {}'.format(e))
                    self.serial.stopbits = backup
                else:
                    if self.logger:
                        self.logger.info('{} stop bits: {}'.format('set' if stopbits else 'get', self.serial.stopbits))
                self.rfc2217_send_subnegotiation(SERVER_SET_STOPSIZE, struct.pack(b'!B', RFC2217_STOPBIT_MAP[self.serial.stopbits]))
            elif suboption[1:2] == SET_CONTROL:
                if suboption[2:3] == SET_CONTROL_REQ_FLOW_SETTING:
                    if self.serial.xonxoff:
                        self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_SW_FLOW_CONTROL)
                    elif self.serial.rtscts:
                        self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_HW_FLOW_CONTROL)
                    else:
                        self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_NO_FLOW_CONTROL)
                elif suboption[2:3] == SET_CONTROL_USE_NO_FLOW_CONTROL:
                    self.serial.xonxoff = False
                    self.serial.rtscts = False
                    if self.logger:
                        self.logger.info('changed flow control to None')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_NO_FLOW_CONTROL)
                elif suboption[2:3] == SET_CONTROL_USE_SW_FLOW_CONTROL:
                    self.serial.xonxoff = True
                    if self.logger:
                        self.logger.info('changed flow control to XON/XOFF')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_SW_FLOW_CONTROL)
                elif suboption[2:3] == SET_CONTROL_USE_HW_FLOW_CONTROL:
                    self.serial.rtscts = True
                    if self.logger:
                        self.logger.info('changed flow control to RTS/CTS')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_USE_HW_FLOW_CONTROL)
                elif suboption[2:3] == SET_CONTROL_REQ_BREAK_STATE:
                    if self.logger:
                        self.logger.warning('requested break state - not implemented')
                    pass
                elif suboption[2:3] == SET_CONTROL_BREAK_ON:
                    self.serial.break_condition = True
                    if self.logger:
                        self.logger.info('changed BREAK to active')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_BREAK_ON)
                elif suboption[2:3] == SET_CONTROL_BREAK_OFF:
                    self.serial.break_condition = False
                    if self.logger:
                        self.logger.info('changed BREAK to inactive')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_BREAK_OFF)
                elif suboption[2:3] == SET_CONTROL_REQ_DTR:
                    if self.logger:
                        self.logger.warning('requested DTR state - not implemented')
                    pass
                elif suboption[2:3] == SET_CONTROL_DTR_ON:
                    self.serial.dtr = True
                    if self.logger:
                        self.logger.info('changed DTR to active')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_DTR_ON)
                elif suboption[2:3] == SET_CONTROL_DTR_OFF:
                    self.serial.dtr = False
                    if self.logger:
                        self.logger.info('changed DTR to inactive')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_DTR_OFF)
                elif suboption[2:3] == SET_CONTROL_REQ_RTS:
                    if self.logger:
                        self.logger.warning('requested RTS state - not implemented')
                    pass
                elif suboption[2:3] == SET_CONTROL_RTS_ON:
                    self.serial.rts = True
                    if self.logger:
                        self.logger.info('changed RTS to active')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_RTS_ON)
                elif suboption[2:3] == SET_CONTROL_RTS_OFF:
                    self.serial.rts = False
                    if self.logger:
                        self.logger.info('changed RTS to inactive')
                    self.rfc2217_send_subnegotiation(SERVER_SET_CONTROL, SET_CONTROL_RTS_OFF)
            elif suboption[1:2] == NOTIFY_LINESTATE:
                self.rfc2217_send_subnegotiation(SERVER_NOTIFY_LINESTATE, to_bytes([0]))
            elif suboption[1:2] == NOTIFY_MODEMSTATE:
                if self.logger:
                    self.logger.info('request for modem state')
                self.check_modem_lines(force_notification=True)
            elif suboption[1:2] == FLOWCONTROL_SUSPEND:
                if self.logger:
                    self.logger.info('suspend')
                self._remote_suspend_flow = True
            elif suboption[1:2] == FLOWCONTROL_RESUME:
                if self.logger:
                    self.logger.info('resume')
                self._remote_suspend_flow = False
            elif suboption[1:2] == SET_LINESTATE_MASK:
                self.linstate_mask = ord(suboption[2:3])
                if self.logger:
                    self.logger.info('line state mask: 0x{:02x}'.format(self.linstate_mask))
            elif suboption[1:2] == SET_MODEMSTATE_MASK:
                self.modemstate_mask = ord(suboption[2:3])
                if self.logger:
                    self.logger.info('modem state mask: 0x{:02x}'.format(self.modemstate_mask))
            elif suboption[1:2] == PURGE_DATA:
                if suboption[2:3] == PURGE_RECEIVE_BUFFER:
                    self.serial.reset_input_buffer()
                    if self.logger:
                        self.logger.info('purge in')
                    self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_RECEIVE_BUFFER)
                elif suboption[2:3] == PURGE_TRANSMIT_BUFFER:
                    self.serial.reset_output_buffer()
                    if self.logger:
                        self.logger.info('purge out')
                    self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_TRANSMIT_BUFFER)
                elif suboption[2:3] == PURGE_BOTH_BUFFERS:
                    self.serial.reset_input_buffer()
                    self.serial.reset_output_buffer()
                    if self.logger:
                        self.logger.info('purge both')
                    self.rfc2217_send_subnegotiation(SERVER_PURGE_DATA, PURGE_BOTH_BUFFERS)
                elif self.logger:
                    self.logger.error('undefined PURGE_DATA: {!r}'.format(list(suboption[2:])))
            elif self.logger:
                self.logger.error('undefined COM_PORT_OPTION: {!r}'.format(list(suboption[1:])))
        elif self.logger:
            self.logger.warning('unknown subnegotiation: {!r}'.format(suboption))
if __name__ == '__main__':
    import sys
    s = Serial('rfc2217://localhost:7000', 115200)
    sys.stdout.write('{}\n'.format(s))
    sys.stdout.write('write...\n')
    s.write(b'hello\n')
    s.flush()
    sys.stdout.write('read: {}\n'.format(s.read(5)))
    s.close()