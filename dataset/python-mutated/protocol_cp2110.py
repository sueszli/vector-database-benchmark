import struct
import threading
try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse
try:
    import Queue
except ImportError:
    import queue as Queue
import hid
import serial
from serial.serialutil import SerialBase, SerialException, PortNotOpenError, to_bytes, Timeout
_REPORT_GETSET_UART_ENABLE = 65
_DISABLE_UART = 0
_ENABLE_UART = 1
_REPORT_SET_PURGE_FIFOS = 67
_PURGE_TX_FIFO = 1
_PURGE_RX_FIFO = 2
_REPORT_GETSET_UART_CONFIG = 80
_REPORT_SET_TRANSMIT_LINE_BREAK = 81
_REPORT_SET_STOP_LINE_BREAK = 82

class Serial(SerialBase):
    BAUDRATES = (300, 375, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200, 230400, 460800, 500000, 576000, 921600, 1000000)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._hid_handle = None
        self._read_buffer = None
        self._thread = None
        super(Serial, self).__init__(*args, **kwargs)

    def open(self):
        if False:
            print('Hello World!')
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if self.is_open:
            raise SerialException('Port is already open.')
        self._read_buffer = Queue.Queue()
        self._hid_handle = hid.device()
        try:
            portpath = self.from_url(self.portstr)
            self._hid_handle.open_path(portpath)
        except OSError as msg:
            raise SerialException(msg.errno, 'could not open port {}: {}'.format(self._port, msg))
        try:
            self._reconfigure_port()
        except:
            try:
                self._hid_handle.close()
            except:
                pass
            self._hid_handle = None
            raise
        else:
            self.is_open = True
            self._thread = threading.Thread(target=self._hid_read_loop)
            self._thread.daemon = True
            self._thread.setName('pySerial CP2110 reader thread for {}'.format(self._port))
            self._thread.start()

    def from_url(self, url):
        if False:
            while True:
                i = 10
        parts = urlparse.urlsplit(url)
        if parts.scheme != 'cp2110':
            raise SerialException('expected a string in the forms "cp2110:///dev/hidraw9" or "cp2110://0001:0023:00": not starting with cp2110:// {{!r}}'.format(parts.scheme))
        if parts.netloc:
            return parts.netloc.encode('utf-8')
        return parts.path.encode('utf-8')

    def close(self):
        if False:
            return 10
        self.is_open = False
        if self._thread:
            self._thread.join(1)
            self._thread = None
        self._hid_handle.close()
        self._hid_handle = None

    def _reconfigure_port(self):
        if False:
            return 10
        parity_value = None
        if self._parity == serial.PARITY_NONE:
            parity_value = 0
        elif self._parity == serial.PARITY_ODD:
            parity_value = 1
        elif self._parity == serial.PARITY_EVEN:
            parity_value = 2
        elif self._parity == serial.PARITY_MARK:
            parity_value = 3
        elif self._parity == serial.PARITY_SPACE:
            parity_value = 4
        else:
            raise ValueError('Invalid parity: {!r}'.format(self._parity))
        if self.rtscts:
            flow_control_value = 1
        else:
            flow_control_value = 0
        data_bits_value = None
        if self._bytesize == 5:
            data_bits_value = 0
        elif self._bytesize == 6:
            data_bits_value = 1
        elif self._bytesize == 7:
            data_bits_value = 2
        elif self._bytesize == 8:
            data_bits_value = 3
        else:
            raise ValueError('Invalid char len: {!r}'.format(self._bytesize))
        stop_bits_value = None
        if self._stopbits == serial.STOPBITS_ONE:
            stop_bits_value = 0
        elif self._stopbits == serial.STOPBITS_ONE_POINT_FIVE:
            stop_bits_value = 1
        elif self._stopbits == serial.STOPBITS_TWO:
            stop_bits_value = 1
        else:
            raise ValueError('Invalid stop bit specification: {!r}'.format(self._stopbits))
        configuration_report = struct.pack('>BLBBBB', _REPORT_GETSET_UART_CONFIG, self._baudrate, parity_value, flow_control_value, data_bits_value, stop_bits_value)
        self._hid_handle.send_feature_report(configuration_report)
        self._hid_handle.send_feature_report(bytes((_REPORT_GETSET_UART_ENABLE, _ENABLE_UART)))
        self._update_break_state()

    @property
    def in_waiting(self):
        if False:
            print('Hello World!')
        return self._read_buffer.qsize()

    def reset_input_buffer(self):
        if False:
            while True:
                i = 10
        if not self.is_open:
            raise PortNotOpenError()
        self._hid_handle.send_feature_report(bytes((_REPORT_SET_PURGE_FIFOS, _PURGE_RX_FIFO)))
        while self._read_buffer.qsize():
            self._read_buffer.get(False)

    def reset_output_buffer(self):
        if False:
            print('Hello World!')
        if not self.is_open:
            raise PortNotOpenError()
        self._hid_handle.send_feature_report(bytes((_REPORT_SET_PURGE_FIFOS, _PURGE_TX_FIFO)))

    def _update_break_state(self):
        if False:
            i = 10
            return i + 15
        if not self._hid_handle:
            raise PortNotOpenError()
        if self._break_state:
            self._hid_handle.send_feature_report(bytes((_REPORT_SET_TRANSMIT_LINE_BREAK, 0)))
        else:
            self._hid_handle.send_feature_report(bytes((_REPORT_SET_STOP_LINE_BREAK, 0)))

    def read(self, size=1):
        if False:
            print('Hello World!')
        if not self.is_open:
            raise PortNotOpenError()
        data = bytearray()
        try:
            timeout = Timeout(self._timeout)
            while len(data) < size:
                if self._thread is None:
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
            print('Hello World!')
        if not self.is_open:
            raise PortNotOpenError()
        data = to_bytes(data)
        tx_len = len(data)
        while tx_len > 0:
            to_be_sent = min(tx_len, 63)
            report = to_bytes([to_be_sent]) + data[:to_be_sent]
            self._hid_handle.write(report)
            data = data[to_be_sent:]
            tx_len = len(data)

    def _hid_read_loop(self):
        if False:
            while True:
                i = 10
        try:
            while self.is_open:
                data = self._hid_handle.read(64, timeout_ms=100)
                if not data:
                    continue
                data_len = data.pop(0)
                assert data_len == len(data)
                self._read_buffer.put(bytearray(data))
        finally:
            self._thread = None