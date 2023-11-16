from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
try:
    import urlparse
except ImportError:
    import urllib.parse as urlparse
from serial.serialutil import SerialBase, SerialException, to_bytes, PortNotOpenError, SerialTimeoutException, Timeout
LOGGER_LEVELS = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING, 'error': logging.ERROR}
POLL_TIMEOUT = 5

class Serial(SerialBase):
    """Serial port implementation for plain sockets."""
    BAUDRATES = (50, 75, 110, 134, 150, 200, 300, 600, 1200, 1800, 2400, 4800, 9600, 19200, 38400, 57600, 115200)

    def open(self):
        if False:
            i = 10
            return i + 15
        '        Open port with current settings. This may throw a SerialException\n        if the port cannot be opened.\n        '
        self.logger = None
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if self.is_open:
            raise SerialException('Port is already open.')
        try:
            self._socket = socket.create_connection(self.from_url(self.portstr), timeout=POLL_TIMEOUT)
        except Exception as msg:
            self._socket = None
            raise SerialException('Could not open port {}: {}'.format(self.portstr, msg))
        self._socket.setblocking(False)
        self._reconfigure_port()
        self.is_open = True
        if not self._dsrdtr:
            self._update_dtr_state()
        if not self._rtscts:
            self._update_rts_state()
        self.reset_input_buffer()
        self.reset_output_buffer()

    def _reconfigure_port(self):
        if False:
            return 10
        '        Set communication parameters on opened port. For the socket://\n        protocol all settings are ignored!\n        '
        if self._socket is None:
            raise SerialException('Can only operate on open ports')
        if self.logger:
            self.logger.info('ignored port configuration change')

    def close(self):
        if False:
            return 10
        'Close port'
        if self.is_open:
            if self._socket:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                    self._socket.close()
                except:
                    pass
                self._socket = None
            self.is_open = False
            time.sleep(0.3)

    def from_url(self, url):
        if False:
            i = 10
            return i + 15
        'extract host and port from an URL string'
        parts = urlparse.urlsplit(url)
        if parts.scheme != 'socket':
            raise SerialException('expected a string in the form "socket://<host>:<port>[?logging={debug|info|warning|error}]": not starting with socket:// ({!r})'.format(parts.scheme))
        try:
            for (option, values) in urlparse.parse_qs(parts.query, True).items():
                if option == 'logging':
                    logging.basicConfig()
                    self.logger = logging.getLogger('pySerial.socket')
                    self.logger.setLevel(LOGGER_LEVELS[values[0]])
                    self.logger.debug('enabled logging')
                else:
                    raise ValueError('unknown option: {!r}'.format(option))
            if not 0 <= parts.port < 65536:
                raise ValueError('port not in range 0...65535')
        except ValueError as e:
            raise SerialException('expected a string in the form "socket://<host>:<port>[?logging={debug|info|warning|error}]": {}'.format(e))
        return (parts.hostname, parts.port)

    @property
    def in_waiting(self):
        if False:
            while True:
                i = 10
        'Return the number of bytes currently in the input buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        (lr, lw, lx) = select.select([self._socket], [], [], 0)
        return len(lr)

    def read(self, size=1):
        if False:
            i = 10
            return i + 15
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        timeout = Timeout(self._timeout)
        while len(read) < size:
            try:
                (ready, _, _) = select.select([self._socket], [], [], timeout.time_left())
                if not ready:
                    break
                buf = self._socket.recv(size - len(read))
                if not buf:
                    raise SerialException('socket disconnected')
                read.extend(buf)
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))
            except (select.error, socket.error) as e:
                if e[0] not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))
            if timeout.expired():
                break
        return bytes(read)

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        '        Output the given byte string over the serial port. Can block if the\n        connection is blocked. May raise SerialException if the connection is\n        closed.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        d = to_bytes(data)
        tx_len = length = len(d)
        timeout = Timeout(self._write_timeout)
        while tx_len > 0:
            try:
                n = self._socket.send(d)
                if timeout.is_non_blocking:
                    return n
                elif not timeout.is_infinite:
                    if timeout.expired():
                        raise SerialTimeoutException('Write timeout')
                    (_, ready, _) = select.select([], [self._socket], [], timeout.time_left())
                    if not ready:
                        raise SerialTimeoutException('Write timeout')
                else:
                    assert timeout.time_left() is None
                    (_, ready, _) = select.select([], [self._socket], [], None)
                    if not ready:
                        raise SerialException('write failed (select)')
                d = d[n:]
                tx_len -= n
            except SerialException:
                raise
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('write failed: {}'.format(e))
            except select.error as e:
                if e[0] not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('write failed: {}'.format(e))
            if not timeout.is_non_blocking and timeout.expired():
                raise SerialTimeoutException('Write timeout')
        return length - len(d)

    def reset_input_buffer(self):
        if False:
            i = 10
            return i + 15
        'Clear input buffer, discarding all that is in the buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        ready = True
        while ready:
            (ready, _, _) = select.select([self._socket], [], [], 0)
            try:
                if ready:
                    ready = self._socket.recv(4096)
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))
            except (select.error, socket.error) as e:
                if e[0] not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))

    def reset_output_buffer(self):
        if False:
            print('Hello World!')
        '        Clear output buffer, aborting the current output and\n        discarding all that is in the buffer.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('ignored reset_output_buffer')

    def send_break(self, duration=0.25):
        if False:
            for i in range(10):
                print('nop')
        '        Send break condition. Timed, returns to idle state after given\n        duration.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('ignored send_break({!r})'.format(duration))

    def _update_break_state(self):
        if False:
            while True:
                i = 10
        'Set break: Controls TXD. When active, to transmitting is\n        possible.'
        if self.logger:
            self.logger.info('ignored _update_break_state({!r})'.format(self._break_state))

    def _update_rts_state(self):
        if False:
            i = 10
            return i + 15
        'Set terminal status line: Request To Send'
        if self.logger:
            self.logger.info('ignored _update_rts_state({!r})'.format(self._rts_state))

    def _update_dtr_state(self):
        if False:
            return 10
        'Set terminal status line: Data Terminal Ready'
        if self.logger:
            self.logger.info('ignored _update_dtr_state({!r})'.format(self._dtr_state))

    @property
    def cts(self):
        if False:
            print('Hello World!')
        'Read terminal status line: Clear To Send'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('returning dummy for cts')
        return True

    @property
    def dsr(self):
        if False:
            while True:
                i = 10
        'Read terminal status line: Data Set Ready'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('returning dummy for dsr')
        return True

    @property
    def ri(self):
        if False:
            while True:
                i = 10
        'Read terminal status line: Ring Indicator'
        if not self.is_open:
            raise PortNotOpenError()
        if self.logger:
            self.logger.info('returning dummy for ri')
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
            self.logger.info('returning dummy for cd)')
        return True

    def fileno(self):
        if False:
            return 10
        'Get the file handle of the underlying socket for use with select'
        return self._socket.fileno()
if __name__ == '__main__':
    import sys
    s = Serial('socket://localhost:7000')
    sys.stdout.write('{}\n'.format(s))
    sys.stdout.write('write...\n')
    s.write(b'hello\n')
    s.flush()
    sys.stdout.write('read: {}\n'.format(s.read(5)))
    s.close()