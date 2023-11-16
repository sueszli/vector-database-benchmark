from __future__ import absolute_import
import errno
import fcntl
import os
import platform
import select
import struct
import sys
import termios
import serial
from serial.serialutil import SerialBase, SerialException, to_bytes, PortNotOpenError, SerialTimeoutException, Timeout

class PlatformSpecificBase(object):
    BAUDRATE_CONSTANTS = {}

    def _set_special_baudrate(self, baudrate):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('non-standard baudrates are not supported on this platform')

    def _set_rs485_mode(self, rs485_settings):
        if False:
            while True:
                i = 10
        raise NotImplementedError('RS485 not supported on this platform')

    def set_low_latency_mode(self, low_latency_settings):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Low latency not supported on this platform')

    def _update_break_state(self):
        if False:
            i = 10
            return i + 15
        '        Set break: Controls TXD. When active, no transmitting is possible.\n        '
        if self._break_state:
            fcntl.ioctl(self.fd, TIOCSBRK)
        else:
            fcntl.ioctl(self.fd, TIOCCBRK)
CMSPAR = 0
plat = sys.platform.lower()
if plat[:5] == 'linux':
    import array
    CMSPAR = 1073741824
    if platform.machine().lower() == 'mips':
        TCGETS2 = 1076909098
        TCSETS2 = 2150650923
        BAUDRATE_OFFSET = 10
    else:
        TCGETS2 = 2150388778
        TCSETS2 = 1076646955
        BAUDRATE_OFFSET = 9
    BOTHER = 4096
    TIOCGRS485 = 21550
    TIOCSRS485 = 21551
    SER_RS485_ENABLED = 1
    SER_RS485_RTS_ON_SEND = 2
    SER_RS485_RTS_AFTER_SEND = 4
    SER_RS485_RX_DURING_TX = 16

    class PlatformSpecific(PlatformSpecificBase):
        BAUDRATE_CONSTANTS = {0: 0, 50: 1, 75: 2, 110: 3, 134: 4, 150: 5, 200: 6, 300: 7, 600: 8, 1200: 9, 1800: 10, 2400: 11, 4800: 12, 9600: 13, 19200: 14, 38400: 15, 57600: 4097, 115200: 4098, 230400: 4099, 460800: 4100, 500000: 4101, 576000: 4102, 921600: 4103, 1000000: 4104, 1152000: 4105, 1500000: 4106, 2000000: 4107, 2500000: 4108, 3000000: 4109, 3500000: 4110, 4000000: 4111}

        def set_low_latency_mode(self, low_latency_settings):
            if False:
                i = 10
                return i + 15
            buf = array.array('i', [0] * 32)
            try:
                fcntl.ioctl(self.fd, termios.TIOCGSERIAL, buf)
                if low_latency_settings:
                    buf[4] |= 8192
                else:
                    buf[4] &= ~8192
                fcntl.ioctl(self.fd, termios.TIOCSSERIAL, buf)
            except IOError as e:
                raise ValueError('Failed to update ASYNC_LOW_LATENCY flag to {}: {}'.format(low_latency_settings, e))

        def _set_special_baudrate(self, baudrate):
            if False:
                for i in range(10):
                    print('nop')
            buf = array.array('i', [0] * 64)
            try:
                fcntl.ioctl(self.fd, TCGETS2, buf)
                buf[2] &= ~termios.CBAUD
                buf[2] |= BOTHER
                buf[BAUDRATE_OFFSET] = buf[BAUDRATE_OFFSET + 1] = baudrate
                fcntl.ioctl(self.fd, TCSETS2, buf)
            except IOError as e:
                raise ValueError('Failed to set custom baud rate ({}): {}'.format(baudrate, e))

        def _set_rs485_mode(self, rs485_settings):
            if False:
                print('Hello World!')
            buf = array.array('i', [0] * 8)
            try:
                fcntl.ioctl(self.fd, TIOCGRS485, buf)
                buf[0] |= SER_RS485_ENABLED
                if rs485_settings is not None:
                    if rs485_settings.loopback:
                        buf[0] |= SER_RS485_RX_DURING_TX
                    else:
                        buf[0] &= ~SER_RS485_RX_DURING_TX
                    if rs485_settings.rts_level_for_tx:
                        buf[0] |= SER_RS485_RTS_ON_SEND
                    else:
                        buf[0] &= ~SER_RS485_RTS_ON_SEND
                    if rs485_settings.rts_level_for_rx:
                        buf[0] |= SER_RS485_RTS_AFTER_SEND
                    else:
                        buf[0] &= ~SER_RS485_RTS_AFTER_SEND
                    if rs485_settings.delay_before_tx is not None:
                        buf[1] = int(rs485_settings.delay_before_tx * 1000)
                    if rs485_settings.delay_before_rx is not None:
                        buf[2] = int(rs485_settings.delay_before_rx * 1000)
                else:
                    buf[0] = 0
                fcntl.ioctl(self.fd, TIOCSRS485, buf)
            except IOError as e:
                raise ValueError('Failed to set RS485 mode: {}'.format(e))
elif plat == 'cygwin':

    class PlatformSpecific(PlatformSpecificBase):
        BAUDRATE_CONSTANTS = {128000: 4099, 256000: 4101, 500000: 4103, 576000: 4104, 921600: 4105, 1000000: 4106, 1152000: 4107, 1500000: 4108, 2000000: 4109, 2500000: 4110, 3000000: 4111}
elif plat[:6] == 'darwin':
    import array
    IOSSIOSPEED = 2147767298

    class PlatformSpecific(PlatformSpecificBase):
        osx_version = os.uname()[2].split('.')
        TIOCSBRK = 536900731
        TIOCCBRK = 536900730
        if int(osx_version[0]) >= 8:

            def _set_special_baudrate(self, baudrate):
                if False:
                    i = 10
                    return i + 15
                buf = array.array('i', [baudrate])
                fcntl.ioctl(self.fd, IOSSIOSPEED, buf, 1)

        def _update_break_state(self):
            if False:
                while True:
                    i = 10
            '            Set break: Controls TXD. When active, no transmitting is possible.\n            '
            if self._break_state:
                fcntl.ioctl(self.fd, PlatformSpecific.TIOCSBRK)
            else:
                fcntl.ioctl(self.fd, PlatformSpecific.TIOCCBRK)
elif plat[:3] == 'bsd' or plat[:7] == 'freebsd' or plat[:6] == 'netbsd' or (plat[:7] == 'openbsd'):

    class ReturnBaudrate(object):

        def __getitem__(self, key):
            if False:
                return 10
            return key

    class PlatformSpecific(PlatformSpecificBase):
        BAUDRATE_CONSTANTS = ReturnBaudrate()
        TIOCSBRK = 536900731
        TIOCCBRK = 536900730

        def _update_break_state(self):
            if False:
                print('Hello World!')
            '            Set break: Controls TXD. When active, no transmitting is possible.\n            '
            if self._break_state:
                fcntl.ioctl(self.fd, PlatformSpecific.TIOCSBRK)
            else:
                fcntl.ioctl(self.fd, PlatformSpecific.TIOCCBRK)
else:

    class PlatformSpecific(PlatformSpecificBase):
        pass
TIOCMGET = getattr(termios, 'TIOCMGET', 21525)
TIOCMBIS = getattr(termios, 'TIOCMBIS', 21526)
TIOCMBIC = getattr(termios, 'TIOCMBIC', 21527)
TIOCMSET = getattr(termios, 'TIOCMSET', 21528)
TIOCM_DTR = getattr(termios, 'TIOCM_DTR', 2)
TIOCM_RTS = getattr(termios, 'TIOCM_RTS', 4)
TIOCM_CTS = getattr(termios, 'TIOCM_CTS', 32)
TIOCM_CAR = getattr(termios, 'TIOCM_CAR', 64)
TIOCM_RNG = getattr(termios, 'TIOCM_RNG', 128)
TIOCM_DSR = getattr(termios, 'TIOCM_DSR', 256)
TIOCM_CD = getattr(termios, 'TIOCM_CD', TIOCM_CAR)
TIOCM_RI = getattr(termios, 'TIOCM_RI', TIOCM_RNG)
if hasattr(termios, 'TIOCINQ'):
    TIOCINQ = termios.TIOCINQ
else:
    TIOCINQ = getattr(termios, 'FIONREAD', 21531)
TIOCOUTQ = getattr(termios, 'TIOCOUTQ', 21521)
TIOCM_zero_str = struct.pack('I', 0)
TIOCM_RTS_str = struct.pack('I', TIOCM_RTS)
TIOCM_DTR_str = struct.pack('I', TIOCM_DTR)
TIOCSBRK = getattr(termios, 'TIOCSBRK', 21543)
TIOCCBRK = getattr(termios, 'TIOCCBRK', 21544)

class Serial(SerialBase, PlatformSpecific):
    """    Serial port class POSIX implementation. Serial port configuration is
    done with termios and fcntl. Runs on Linux and many other Un*x like
    systems.
    """

    def open(self):
        if False:
            while True:
                i = 10
        '        Open port with current settings. This may throw a SerialException\n        if the port cannot be opened.'
        if self._port is None:
            raise SerialException('Port must be configured before it can be used.')
        if self.is_open:
            raise SerialException('Port is already open.')
        self.fd = None
        try:
            self.fd = os.open(self.portstr, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as msg:
            self.fd = None
            raise SerialException(msg.errno, 'could not open port {}: {}'.format(self._port, msg))
        (self.pipe_abort_read_r, self.pipe_abort_read_w) = (None, None)
        (self.pipe_abort_write_r, self.pipe_abort_write_w) = (None, None)
        try:
            self._reconfigure_port(force_update=True)
            try:
                if not self._dsrdtr:
                    self._update_dtr_state()
                if not self._rtscts:
                    self._update_rts_state()
            except IOError as e:
                if e.errno not in (errno.EINVAL, errno.ENOTTY):
                    raise
            self._reset_input_buffer()
            (self.pipe_abort_read_r, self.pipe_abort_read_w) = os.pipe()
            (self.pipe_abort_write_r, self.pipe_abort_write_w) = os.pipe()
            fcntl.fcntl(self.pipe_abort_read_r, fcntl.F_SETFL, os.O_NONBLOCK)
            fcntl.fcntl(self.pipe_abort_write_r, fcntl.F_SETFL, os.O_NONBLOCK)
        except BaseException:
            try:
                os.close(self.fd)
            except Exception:
                pass
            self.fd = None
            if self.pipe_abort_read_w is not None:
                os.close(self.pipe_abort_read_w)
                self.pipe_abort_read_w = None
            if self.pipe_abort_read_r is not None:
                os.close(self.pipe_abort_read_r)
                self.pipe_abort_read_r = None
            if self.pipe_abort_write_w is not None:
                os.close(self.pipe_abort_write_w)
                self.pipe_abort_write_w = None
            if self.pipe_abort_write_r is not None:
                os.close(self.pipe_abort_write_r)
                self.pipe_abort_write_r = None
            raise
        self.is_open = True

    def _reconfigure_port(self, force_update=False):
        if False:
            i = 10
            return i + 15
        'Set communication parameters on opened port.'
        if self.fd is None:
            raise SerialException('Can only operate on a valid file descriptor')
        if self._exclusive is not None:
            if self._exclusive:
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError as msg:
                    raise SerialException(msg.errno, 'Could not exclusively lock port {}: {}'.format(self._port, msg))
            else:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
        custom_baud = None
        vmin = vtime = 0
        if self._inter_byte_timeout is not None:
            vmin = 1
            vtime = int(self._inter_byte_timeout * 10)
        try:
            orig_attr = termios.tcgetattr(self.fd)
            (iflag, oflag, cflag, lflag, ispeed, ospeed, cc) = orig_attr
        except termios.error as msg:
            raise SerialException('Could not configure port: {}'.format(msg))
        cflag |= termios.CLOCAL | termios.CREAD
        lflag &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ECHOK | termios.ECHONL | termios.ISIG | termios.IEXTEN)
        for flag in ('ECHOCTL', 'ECHOKE'):
            if hasattr(termios, flag):
                lflag &= ~getattr(termios, flag)
        oflag &= ~(termios.OPOST | termios.ONLCR | termios.OCRNL)
        iflag &= ~(termios.INLCR | termios.IGNCR | termios.ICRNL | termios.IGNBRK)
        if hasattr(termios, 'IUCLC'):
            iflag &= ~termios.IUCLC
        if hasattr(termios, 'PARMRK'):
            iflag &= ~termios.PARMRK
        try:
            ispeed = ospeed = getattr(termios, 'B{}'.format(self._baudrate))
        except AttributeError:
            try:
                ispeed = ospeed = self.BAUDRATE_CONSTANTS[self._baudrate]
            except KeyError:
                try:
                    ispeed = ospeed = BOTHER
                except NameError:
                    ispeed = ospeed = getattr(termios, 'B38400')
                try:
                    custom_baud = int(self._baudrate)
                except ValueError:
                    raise ValueError('Invalid baud rate: {!r}'.format(self._baudrate))
                else:
                    if custom_baud < 0:
                        raise ValueError('Invalid baud rate: {!r}'.format(self._baudrate))
        cflag &= ~termios.CSIZE
        if self._bytesize == 8:
            cflag |= termios.CS8
        elif self._bytesize == 7:
            cflag |= termios.CS7
        elif self._bytesize == 6:
            cflag |= termios.CS6
        elif self._bytesize == 5:
            cflag |= termios.CS5
        else:
            raise ValueError('Invalid char len: {!r}'.format(self._bytesize))
        if self._stopbits == serial.STOPBITS_ONE:
            cflag &= ~termios.CSTOPB
        elif self._stopbits == serial.STOPBITS_ONE_POINT_FIVE:
            cflag |= termios.CSTOPB
        elif self._stopbits == serial.STOPBITS_TWO:
            cflag |= termios.CSTOPB
        else:
            raise ValueError('Invalid stop bit specification: {!r}'.format(self._stopbits))
        iflag &= ~(termios.INPCK | termios.ISTRIP)
        if self._parity == serial.PARITY_NONE:
            cflag &= ~(termios.PARENB | termios.PARODD | CMSPAR)
        elif self._parity == serial.PARITY_EVEN:
            cflag &= ~(termios.PARODD | CMSPAR)
            cflag |= termios.PARENB
        elif self._parity == serial.PARITY_ODD:
            cflag &= ~CMSPAR
            cflag |= termios.PARENB | termios.PARODD
        elif self._parity == serial.PARITY_MARK and CMSPAR:
            cflag |= termios.PARENB | CMSPAR | termios.PARODD
        elif self._parity == serial.PARITY_SPACE and CMSPAR:
            cflag |= termios.PARENB | CMSPAR
            cflag &= ~termios.PARODD
        else:
            raise ValueError('Invalid parity: {!r}'.format(self._parity))
        if hasattr(termios, 'IXANY'):
            if self._xonxoff:
                iflag |= termios.IXON | termios.IXOFF
            else:
                iflag &= ~(termios.IXON | termios.IXOFF | termios.IXANY)
        elif self._xonxoff:
            iflag |= termios.IXON | termios.IXOFF
        else:
            iflag &= ~(termios.IXON | termios.IXOFF)
        if hasattr(termios, 'CRTSCTS'):
            if self._rtscts:
                cflag |= termios.CRTSCTS
            else:
                cflag &= ~termios.CRTSCTS
        elif hasattr(termios, 'CNEW_RTSCTS'):
            if self._rtscts:
                cflag |= termios.CNEW_RTSCTS
            else:
                cflag &= ~termios.CNEW_RTSCTS
        if vmin < 0 or vmin > 255:
            raise ValueError('Invalid vmin: {!r}'.format(vmin))
        cc[termios.VMIN] = vmin
        if vtime < 0 or vtime > 255:
            raise ValueError('Invalid vtime: {!r}'.format(vtime))
        cc[termios.VTIME] = vtime
        if force_update or [iflag, oflag, cflag, lflag, ispeed, ospeed, cc] != orig_attr:
            termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc])
        if custom_baud is not None:
            self._set_special_baudrate(custom_baud)
        if self._rs485_mode is not None:
            self._set_rs485_mode(self._rs485_mode)

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close port'
        if self.is_open:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
                os.close(self.pipe_abort_read_w)
                os.close(self.pipe_abort_read_r)
                os.close(self.pipe_abort_write_w)
                os.close(self.pipe_abort_write_r)
                (self.pipe_abort_read_r, self.pipe_abort_read_w) = (None, None)
                (self.pipe_abort_write_r, self.pipe_abort_write_w) = (None, None)
            self.is_open = False

    @property
    def in_waiting(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the number of bytes currently in the input buffer.'
        s = fcntl.ioctl(self.fd, TIOCINQ, TIOCM_zero_str)
        return struct.unpack('I', s)[0]

    def read(self, size=1):
        if False:
            return 10
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        timeout = Timeout(self._timeout)
        while len(read) < size:
            try:
                (ready, _, _) = select.select([self.fd, self.pipe_abort_read_r], [], [], timeout.time_left())
                if self.pipe_abort_read_r in ready:
                    os.read(self.pipe_abort_read_r, 1000)
                    break
                if not ready:
                    break
                buf = os.read(self.fd, size - len(read))
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))
            except select.error as e:
                if e[0] not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                    raise SerialException('read failed: {}'.format(e))
            else:
                if not buf:
                    raise SerialException('device reports readiness to read but returned no data (device disconnected or multiple access on port?)')
                read.extend(buf)
            if timeout.expired():
                break
        return bytes(read)

    def cancel_read(self):
        if False:
            i = 10
            return i + 15
        if self.is_open:
            os.write(self.pipe_abort_read_w, b'x')

    def cancel_write(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_open:
            os.write(self.pipe_abort_write_w, b'x')

    def write(self, data):
        if False:
            return 10
        'Output the given byte string over the serial port.'
        if not self.is_open:
            raise PortNotOpenError()
        d = to_bytes(data)
        tx_len = length = len(d)
        timeout = Timeout(self._write_timeout)
        while tx_len > 0:
            try:
                n = os.write(self.fd, d)
                if timeout.is_non_blocking:
                    return n
                elif not timeout.is_infinite:
                    if timeout.expired():
                        raise SerialTimeoutException('Write timeout')
                    (abort, ready, _) = select.select([self.pipe_abort_write_r], [self.fd], [], timeout.time_left())
                    if abort:
                        os.read(self.pipe_abort_write_r, 1000)
                        break
                    if not ready:
                        raise SerialTimeoutException('Write timeout')
                else:
                    assert timeout.time_left() is None
                    (abort, ready, _) = select.select([self.pipe_abort_write_r], [self.fd], [], None)
                    if abort:
                        os.read(self.pipe_abort_write_r, 1)
                        break
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

    def flush(self):
        if False:
            print('Hello World!')
        '        Flush of file like objects. In this case, wait until all data\n        is written.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        termios.tcdrain(self.fd)

    def _reset_input_buffer(self):
        if False:
            print('Hello World!')
        'Clear input buffer, discarding all that is in the buffer.'
        termios.tcflush(self.fd, termios.TCIFLUSH)

    def reset_input_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear input buffer, discarding all that is in the buffer.'
        if not self.is_open:
            raise PortNotOpenError()
        self._reset_input_buffer()

    def reset_output_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        '        Clear output buffer, aborting the current output and discarding all\n        that is in the buffer.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        termios.tcflush(self.fd, termios.TCOFLUSH)

    def send_break(self, duration=0.25):
        if False:
            while True:
                i = 10
        '        Send break condition. Timed, returns to idle state after given\n        duration.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        termios.tcsendbreak(self.fd, int(duration / 0.25))

    def _update_rts_state(self):
        if False:
            for i in range(10):
                print('nop')
        'Set terminal status line: Request To Send'
        if self._rts_state:
            fcntl.ioctl(self.fd, TIOCMBIS, TIOCM_RTS_str)
        else:
            fcntl.ioctl(self.fd, TIOCMBIC, TIOCM_RTS_str)

    def _update_dtr_state(self):
        if False:
            return 10
        'Set terminal status line: Data Terminal Ready'
        if self._dtr_state:
            fcntl.ioctl(self.fd, TIOCMBIS, TIOCM_DTR_str)
        else:
            fcntl.ioctl(self.fd, TIOCMBIC, TIOCM_DTR_str)

    @property
    def cts(self):
        if False:
            while True:
                i = 10
        'Read terminal status line: Clear To Send'
        if not self.is_open:
            raise PortNotOpenError()
        s = fcntl.ioctl(self.fd, TIOCMGET, TIOCM_zero_str)
        return struct.unpack('I', s)[0] & TIOCM_CTS != 0

    @property
    def dsr(self):
        if False:
            for i in range(10):
                print('nop')
        'Read terminal status line: Data Set Ready'
        if not self.is_open:
            raise PortNotOpenError()
        s = fcntl.ioctl(self.fd, TIOCMGET, TIOCM_zero_str)
        return struct.unpack('I', s)[0] & TIOCM_DSR != 0

    @property
    def ri(self):
        if False:
            return 10
        'Read terminal status line: Ring Indicator'
        if not self.is_open:
            raise PortNotOpenError()
        s = fcntl.ioctl(self.fd, TIOCMGET, TIOCM_zero_str)
        return struct.unpack('I', s)[0] & TIOCM_RI != 0

    @property
    def cd(self):
        if False:
            i = 10
            return i + 15
        'Read terminal status line: Carrier Detect'
        if not self.is_open:
            raise PortNotOpenError()
        s = fcntl.ioctl(self.fd, TIOCMGET, TIOCM_zero_str)
        return struct.unpack('I', s)[0] & TIOCM_CD != 0

    @property
    def out_waiting(self):
        if False:
            while True:
                i = 10
        'Return the number of bytes currently in the output buffer.'
        s = fcntl.ioctl(self.fd, TIOCOUTQ, TIOCM_zero_str)
        return struct.unpack('I', s)[0]

    def fileno(self):
        if False:
            i = 10
            return i + 15
        '        For easier use of the serial port instance with select.\n        WARNING: this function is not portable to different platforms!\n        '
        if not self.is_open:
            raise PortNotOpenError()
        return self.fd

    def set_input_flow_control(self, enable=True):
        if False:
            for i in range(10):
                print('nop')
        '        Manually control flow - when software flow control is enabled.\n        This will send XON (true) or XOFF (false) to the other device.\n        WARNING: this function is not portable to different platforms!\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if enable:
            termios.tcflow(self.fd, termios.TCION)
        else:
            termios.tcflow(self.fd, termios.TCIOFF)

    def set_output_flow_control(self, enable=True):
        if False:
            return 10
        '        Manually control flow of outgoing data - when hardware or software flow\n        control is enabled.\n        WARNING: this function is not portable to different platforms!\n        '
        if not self.is_open:
            raise PortNotOpenError()
        if enable:
            termios.tcflow(self.fd, termios.TCOON)
        else:
            termios.tcflow(self.fd, termios.TCOOFF)

    def nonblocking(self):
        if False:
            while True:
                i = 10
        'DEPRECATED - has no use'
        import warnings
        warnings.warn('nonblocking() has no effect, already nonblocking', DeprecationWarning)

class PosixPollSerial(Serial):
    """    Poll based read implementation. Not all systems support poll properly.
    However this one has better handling of errors, such as a device
    disconnecting while it's in use (e.g. USB-serial unplugged).
    """

    def read(self, size=1):
        if False:
            while True:
                i = 10
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        timeout = Timeout(self._timeout)
        poll = select.poll()
        poll.register(self.fd, select.POLLIN | select.POLLERR | select.POLLHUP | select.POLLNVAL)
        poll.register(self.pipe_abort_read_r, select.POLLIN | select.POLLERR | select.POLLHUP | select.POLLNVAL)
        if size > 0:
            while len(read) < size:
                for (fd, event) in poll.poll(None if timeout.is_infinite else timeout.time_left() * 1000):
                    if fd == self.pipe_abort_read_r:
                        break
                    if event & (select.POLLERR | select.POLLHUP | select.POLLNVAL):
                        raise SerialException('device reports error (poll)')
                if fd == self.pipe_abort_read_r:
                    os.read(self.pipe_abort_read_r, 1000)
                    break
                buf = os.read(self.fd, size - len(read))
                read.extend(buf)
                if timeout.expired() or ((self._inter_byte_timeout is not None and self._inter_byte_timeout > 0) and (not buf)):
                    break
        return bytes(read)

class VTIMESerial(Serial):
    """    Implement timeout using vtime of tty device instead of using select.
    This means that no inter character timeout can be specified and that
    the error handling is degraded.

    Overall timeout is disabled when inter-character timeout is used.

    Note that this implementation does NOT support cancel_read(), it will
    just ignore that.
    """

    def _reconfigure_port(self, force_update=True):
        if False:
            return 10
        'Set communication parameters on opened port.'
        super(VTIMESerial, self)._reconfigure_port()
        fcntl.fcntl(self.fd, fcntl.F_SETFL, 0)
        if self._inter_byte_timeout is not None:
            vmin = 1
            vtime = int(self._inter_byte_timeout * 10)
        elif self._timeout is None:
            vmin = 1
            vtime = 0
        else:
            vmin = 0
            vtime = int(self._timeout * 10)
        try:
            orig_attr = termios.tcgetattr(self.fd)
            (iflag, oflag, cflag, lflag, ispeed, ospeed, cc) = orig_attr
        except termios.error as msg:
            raise serial.SerialException('Could not configure port: {}'.format(msg))
        if vtime < 0 or vtime > 255:
            raise ValueError('Invalid vtime: {!r}'.format(vtime))
        cc[termios.VTIME] = vtime
        cc[termios.VMIN] = vmin
        termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, ispeed, ospeed, cc])

    def read(self, size=1):
        if False:
            while True:
                i = 10
        '        Read size bytes from the serial port. If a timeout is set it may\n        return less characters as requested. With no timeout it will block\n        until the requested number of bytes is read.\n        '
        if not self.is_open:
            raise PortNotOpenError()
        read = bytearray()
        while len(read) < size:
            buf = os.read(self.fd, size - len(read))
            if not buf:
                break
            read.extend(buf)
        return bytes(read)
    cancel_read = property()