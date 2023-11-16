from __future__ import absolute_import
import logging
import numbers
import time
try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse
try:
    import queue
except ImportError:
    import Queue as queue
from serial.serialutil import SerialBase, SerialException, to_bytes, iterbytes, SerialTimeoutException, PortNotOpenError
LOGGER_LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}

class Serial(SerialBase):
    """Serial port implementation that simulates a loop back connection in plain software."""
    BAUDRATES = (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.buffer_size = 4096
        self.queue = None
        self.logger = None
        self._cancel_write = False
        super(Serial, self).__init__(*args, **kwargs)

    def open(self):
        if False:
            return 10
        '        Open port with current settings. This may throw a SerialException\n        if the port cannot be opened.\n        '
        if self.is_open:
            raise SerialException('Port is already open.')
        self.logger = None
        self.queue = queue.Queue(self.buffer_size)
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        self.from_url(self.port)
        self._reconfigure_port()
        self.is_open = True
        if not self._dsrdtr:
            self._update_dtr_state()
        if not self._rtscts:
            self._update_rts_state()
        self.reset_input_buffer()
        self.reset_output_buffer()

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_open:
            self.is_open = False
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
        super(Serial, self).close()

    def _reconfigure_port(self):
        if False:
            return 10
        '        Set communication parameters on opened port. For the loop://\n        protocol all settings are ignored!\n        '
        if not isinstance(self._baudrate, numbers.Integral) or not 0 < self._baudrate < 2 ** 32:
            raise ValueError('invalid baudrate: {!r}'.format(self._baudrate))
        if self.logger:
            self.logger.info('_reconfigure_port()')

    def from_url(self, url):
        if False:
            i = 10
            return i + 15
        'extract host and port from an URL string'
        parts = urlparse.urlsplit(url)
        if parts.scheme != 'loop':
            raise SerialException('expected a string in the form "loop://[?logging={debug|info|warning|error}]": not starting with loop:// ({!r})'.format(parts.scheme))
        try:
            for (option, values) in urlparse.parse_qs(parts.query, True).items():
                if option == 'logging':
                    logging.basicConfig()
                    self.logger = logging.getLogger('pySerial.loop')
                    self.logger.setLevel(LOGGER_LEVELS[values[0]])
                    self.logger.debug('enabled logging')
                else:
                    raise ValueError('unknown option: {!r}'.format(option))
        except ValueError as e:
            raise SerialException('expected a string in the form "loop://[?logging={debug|info|warning|error}]": {}'.format(e))

    @property
    def in_waiting(self):
        if False:
            print('Hello World!')
        'Return the number of bytes currently in the input buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.debug('in_waiting -> {:d}'.format(self.queue.qsize()))
        return self.queue.qsize()

    def read(self, size=1):
        if False:
            i = 10
            return i + 15
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if self._timeout is not None and self._timeout != 0:
            timeout = time.time() + self._timeout
        else:
            timeout = None
        data = bytearray()
        while size > 0 and self.is_open:
            try:
                b = self.queue.get(timeout=self._timeout)
            except queue.Empty:
                if self._timeout == 0:
                    break
            else:
                if b is not None:
                    data += b
                    size -= 1
                else:
                    break
            if timeout and time.time() > timeout:
                if self.logger:
                    self.logger.info('read timeout')
                break
        return bytes(data)

    def cancel_read(self):
        if False:
            return 10
        self.queue.put_nowait(None)

    def cancel_write(self):
        if False:
            for i in range(10):
                print('nop')
        self._cancel_write = True

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        '        Output the given byte string over the serial port. Can block if the\n        connection is blocked. May raise SerialException if the connection is\n        closed.\n        '
        self._cancel_write = False
        if not self.is_open:
            raise PortNotOpenError()
        data = to_bytes(data)
        time_used_to_send = 10.0 * len(data) / self._baudrate
        if self._write_timeout is not None and time_used_to_send > self._write_timeout:
            time_left = self._write_timeout
            while time_left > 0 and (not self._cancel_write):
                time.sleep(min(time_left, 0.5))
                time_left -= 0.5
            if self._cancel_write:
                return 0
            raise SerialTimeoutException('Write timeout')
        for byte in iterbytes(data):
            self.queue.put(byte, timeout=self._write_timeout)
        return len(data)

    def reset_input_buffer(self):
        if False:
            return 10
        'Clear input buffer, discarding all that is in the buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('reset_input_buffer()')
        try:
            while self.queue.qsize():
                self.queue.get_nowait()
        except queue.Empty:
            pass

    def reset_output_buffer(self):
        if False:
            i = 10
            return i + 15
        '        Clear output buffer, aborting the current output and\n        discarding all that is in the buffer.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('reset_output_buffer()')
        try:
            while self.queue.qsize():
                self.queue.get_nowait()
        except queue.Empty:
            pass

    @property
    def out_waiting(self):
        if False:
            while True:
                i = 10
        'Return how many bytes the in the outgoing buffer'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.debug('out_waiting -> {:d}'.format(self.queue.qsize()))
        return self.queue.qsize()

    def _update_break_state(self):
        if False:
            i = 10
            return i + 15
        '        Set break: Controls TXD. When active, to transmitting is\n        possible.\n        '
        if self.logger:
            self.logger.info('_update_break_state({!r})'.format(self._break_state))

    def _update_rts_state(self):
        if False:
            return 10
        'Set terminal status line: Request To Send'
        if self.logger:
            self.logger.info('_update_rts_state({!r}) -> state of CTS'.format(self._rts_state))

    def _update_dtr_state(self):
        if False:
            while True:
                i = 10
        'Set terminal status line: Data Terminal Ready'
        if self.logger:
            self.logger.info('_update_dtr_state({!r}) -> state of DSR'.format(self._dtr_state))

    @property
    def cts(self):
        if False:
            print('Hello World!')
        'Read terminal status line: Clear To Send'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('CTS -> state of RTS ({!r})'.format(self._rts_state))
        return self._rts_state

    @property
    def dsr(self):
        if False:
            while True:
                i = 10
        'Read terminal status line: Data Set Ready'
        if self.logger:
            self.logger.info('DSR -> state of DTR ({!r})'.format(self._dtr_state))
        return self._dtr_state

    @property
    def ri(self):
        if False:
            i = 10
            return i + 15
        'Read terminal status line: Ring Indicator'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('returning dummy for RI')
        return False

    @property
    def cd(self):
        if False:
            i = 10
            return i + 15
        'Read terminal status line: Carrier Detect'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('returning dummy for CD')
        return True
if __name__ == '__main__':
    import sys
    s = Serial('loop://')
    sys.stdout.write('{}\n'.format(s))
    sys.stdout.write('write...\n')
    s.write('hello\n')
    s.flush()
    sys.stdout.write('read: {!r}\n'.format(s.read(5)))
    s.close()