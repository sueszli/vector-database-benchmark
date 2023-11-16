import logging.handlers as handlers, socket, os, time

class WebsockifySysLogHandler(handlers.SysLogHandler):
    """
    A handler class that sends proper Syslog-formatted messages,
    as defined by RFC 5424.
    """
    _legacy_head_fmt = '<{pri}>{ident}[{pid}]: '
    _rfc5424_head_fmt = '<{pri}>1 {timestamp} {hostname} {ident} {pid} - - '
    _head_fmt = _rfc5424_head_fmt
    _legacy = False
    _timestamp_fmt = '%Y-%m-%dT%H:%M:%SZ'
    _max_hostname = 255
    _max_ident = 24
    _send_length = False
    _tail = '\n'
    ident = None

    def __init__(self, address=('localhost', handlers.SYSLOG_UDP_PORT), facility=handlers.SysLogHandler.LOG_USER, socktype=None, ident=None, legacy=False):
        if False:
            i = 10
            return i + 15
        '\n        Initialize a handler.\n\n        If address is specified as a string, a UNIX socket is used. To log to a\n        local syslogd, "WebsockifySysLogHandler(address="/dev/log")" can be\n        used. If facility is not specified, LOG_USER is used. If socktype is\n        specified as socket.SOCK_DGRAM or socket.SOCK_STREAM, that specific\n        socket type will be used. For Unix sockets, you can also specify a\n        socktype of None, in which case socket.SOCK_DGRAM will be used, falling\n        back to socket.SOCK_STREAM. If ident is specified, this string will be\n        used as the application name in all messages sent. Set legacy to True\n        to use the old version of the protocol.\n        '
        self.ident = ident
        if legacy:
            self._legacy = True
            self._head_fmt = self._legacy_head_fmt
        super().__init__(address, facility, socktype)

    def emit(self, record):
        if False:
            return 10
        '\n        Emit a record.\n\n        The record is formatted, and then sent to the syslog server. If\n        exception information is present, it is NOT sent to the server.\n        '
        try:
            text = self.format(record).replace(self._tail, ' ')
            if not text:
                return
            pri = self.encodePriority(self.facility, self.mapPriority(record.levelname))
            timestamp = time.strftime(self._timestamp_fmt, time.gmtime())
            hostname = socket.gethostname()[:self._max_hostname]
            if self.ident:
                ident = self.ident[:self._max_ident]
            else:
                ident = ''
            pid = os.getpid()
            head = {'pri': pri, 'timestamp': timestamp, 'hostname': hostname, 'ident': ident, 'pid': pid}
            msg = self._head_fmt.format(**head).encode('ascii', 'ignore')
            try:
                msg += text.encode('ascii')
            except UnicodeEncodeError:
                msg += text.encode('utf-8-sig')
            if self.socktype != socket.SOCK_DGRAM:
                if self._send_length:
                    msg = ('%d ' % len(msg)).encode('ascii') + msg
                else:
                    msg += self._tail.encode('ascii')
            if self.unixsocket:
                try:
                    self.socket.send(msg)
                except socket.error:
                    self._connect_unixsocket(self.address)
                    self.socket.send(msg)
            elif self.socktype == socket.SOCK_DGRAM:
                self.socket.sendto(msg, self.address)
            else:
                self.socket.sendall(msg)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)