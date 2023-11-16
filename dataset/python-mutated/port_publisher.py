"""Multi-port serial<->TCP/IP forwarder.
- RFC 2217
- check existence of serial port periodically
- start/stop forwarders
- each forwarder creates a server socket and opens the serial port
- serial ports are opened only once. network connect/disconnect
  does not influence serial port
- only one client per connection
"""
import os
import select
import socket
import sys
import time
import traceback
import serial
import serial.rfc2217
import serial.tools.list_ports
import dbus
try:
    import avahi
except ImportError:

    class avahi:
        DBUS_NAME = 'org.freedesktop.Avahi'
        DBUS_PATH_SERVER = '/'
        DBUS_INTERFACE_SERVER = 'org.freedesktop.Avahi.Server'
        DBUS_INTERFACE_ENTRY_GROUP = DBUS_NAME + '.EntryGroup'
        IF_UNSPEC = -1
        (PROTO_UNSPEC, PROTO_INET, PROTO_INET6) = (-1, 0, 1)

class ZeroconfService:
    """    A simple class to publish a network service with zeroconf using avahi.
    """

    def __init__(self, name, port, stype='_http._tcp', domain='', host='', text=''):
        if False:
            print('Hello World!')
        self.name = name
        self.stype = stype
        self.domain = domain
        self.host = host
        self.port = port
        self.text = text
        self.group = None

    def publish(self):
        if False:
            while True:
                i = 10
        bus = dbus.SystemBus()
        server = dbus.Interface(bus.get_object(avahi.DBUS_NAME, avahi.DBUS_PATH_SERVER), avahi.DBUS_INTERFACE_SERVER)
        g = dbus.Interface(bus.get_object(avahi.DBUS_NAME, server.EntryGroupNew()), avahi.DBUS_INTERFACE_ENTRY_GROUP)
        g.AddService(avahi.IF_UNSPEC, avahi.PROTO_UNSPEC, dbus.UInt32(0), self.name, self.stype, self.domain, self.host, dbus.UInt16(self.port), self.text)
        g.Commit()
        self.group = g

    def unpublish(self):
        if False:
            i = 10
            return i + 15
        if self.group is not None:
            self.group.Reset()
            self.group = None

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{!r} @ {}:{} ({})'.format(self.name, self.host, self.port, self.stype)

class Forwarder(ZeroconfService):
    """    Single port serial<->TCP/IP forarder that depends on an external select
    loop.
    - Buffers for serial -> network and network -> serial
    - RFC 2217 state
    - Zeroconf publish/unpublish on open/close.
    """

    def __init__(self, device, name, network_port, on_close=None, log=None):
        if False:
            print('Hello World!')
        ZeroconfService.__init__(self, name, network_port, stype='_serial_port._tcp')
        self.alive = False
        self.network_port = network_port
        self.on_close = on_close
        self.log = log
        self.device = device
        self.serial = serial.Serial()
        self.serial.port = device
        self.serial.baudrate = 115200
        self.serial.timeout = 0
        self.socket = None
        self.server_socket = None
        self.rfc2217 = None

    def __del__(self):
        if False:
            return 10
        try:
            if self.alive:
                self.close()
        except:
            pass

    def open(self):
        if False:
            while True:
                i = 10
        'open serial port, start network server and publish service'
        self.buffer_net2ser = bytearray()
        self.buffer_ser2net = bytearray()
        try:
            self.serial.rts = False
            self.serial.open()
        except Exception as msg:
            self.handle_serial_error(msg)
        self.serial_settings_backup = self.serial.get_settings()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, self.server_socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR) | 1)
        self.server_socket.setblocking(0)
        try:
            self.server_socket.bind(('', self.network_port))
            self.server_socket.listen(1)
        except socket.error as msg:
            self.handle_server_error()
        if self.log is not None:
            self.log.info('{}: Waiting for connection on {}...'.format(self.device, self.network_port))
        self.publish()
        self.alive = True

    def close(self):
        if False:
            print('Hello World!')
        'Close all resources and unpublish service'
        if self.log is not None:
            self.log.info('{}: closing...'.format(self.device))
        self.alive = False
        self.unpublish()
        if self.server_socket:
            self.server_socket.close()
        if self.socket:
            self.handle_disconnect()
        self.serial.close()
        if self.on_close is not None:
            callback = self.on_close
            self.on_close = None
            callback(self)

    def write(self, data):
        if False:
            for i in range(10):
                print('nop')
        'the write method is used by serial.rfc2217.PortManager. it has to\n        write to the network.'
        self.buffer_ser2net += data

    def update_select_maps(self, read_map, write_map, error_map):
        if False:
            i = 10
            return i + 15
        'Update dictionaries for select call. insert fd->callback mapping'
        if self.alive:
            read_map[self.serial] = self.handle_serial_read
            error_map[self.serial] = self.handle_serial_error
            if self.buffer_net2ser:
                write_map[self.serial] = self.handle_serial_write
            if self.socket is not None:
                if len(self.buffer_net2ser) < 2048:
                    read_map[self.socket] = self.handle_socket_read
                if self.buffer_ser2net:
                    write_map[self.socket] = self.handle_socket_write
                error_map[self.socket] = self.handle_socket_error
            else:
                self.buffer_ser2net = bytearray()
            read_map[self.server_socket] = self.handle_connect
            error_map[self.server_socket] = self.handle_server_error

    def handle_serial_read(self):
        if False:
            for i in range(10):
                print('nop')
        'Reading from serial port'
        try:
            data = os.read(self.serial.fileno(), 1024)
            if data:
                if self.socket is not None:
                    if self.rfc2217:
                        data = serial.to_bytes(self.rfc2217.escape(data))
                    self.buffer_ser2net.extend(data)
            else:
                self.handle_serial_error()
        except Exception as msg:
            self.handle_serial_error(msg)

    def handle_serial_write(self):
        if False:
            for i in range(10):
                print('nop')
        'Writing to serial port'
        try:
            n = os.write(self.serial.fileno(), bytes(self.buffer_net2ser))
            self.buffer_net2ser = self.buffer_net2ser[n:]
        except Exception as msg:
            self.handle_serial_error(msg)

    def handle_serial_error(self, error=None):
        if False:
            return 10
        'Serial port error'
        self.close()

    def handle_socket_read(self):
        if False:
            for i in range(10):
                print('nop')
        'Read from socket'
        try:
            data = self.socket.recv(1024)
            if data:
                if self.rfc2217:
                    data = b''.join(self.rfc2217.filter(data))
                self.buffer_net2ser.extend(data)
            else:
                self.handle_disconnect()
        except socket.error:
            if self.log is not None:
                self.log.exception('{}: error reading...'.format(self.device))
            self.handle_socket_error()

    def handle_socket_write(self):
        if False:
            while True:
                i = 10
        'Write to socket'
        try:
            count = self.socket.send(bytes(self.buffer_ser2net))
            self.buffer_ser2net = self.buffer_ser2net[count:]
        except socket.error:
            if self.log is not None:
                self.log.exception('{}: error writing...'.format(self.device))
            self.handle_socket_error()

    def handle_socket_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Socket connection fails'
        self.handle_disconnect()

    def handle_connect(self):
        if False:
            while True:
                i = 10
        'Server socket gets a connection'
        (connection, addr) = self.server_socket.accept()
        if self.socket is None:
            self.socket = connection
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            self.socket.setblocking(0)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.log is not None:
                self.log.warning('{}: Connected by {}:{}'.format(self.device, addr[0], addr[1]))
            self.serial.rts = True
            self.serial.dtr = True
            if self.log is not None:
                self.rfc2217 = serial.rfc2217.PortManager(self.serial, self, logger=log.getChild(self.device))
            else:
                self.rfc2217 = serial.rfc2217.PortManager(self.serial, self)
        else:
            connection.close()
            if self.log is not None:
                self.log.warning('{}: Rejecting connect from {}:{}'.format(self.device, addr[0], addr[1]))

    def handle_server_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Socket server fails'
        self.close()

    def handle_disconnect(self):
        if False:
            return 10
        'Socket gets disconnected'
        try:
            self.serial.rts = False
            self.serial.dtr = False
        finally:
            self.serial.apply_settings(self.serial_settings_backup)
            self.rfc2217 = None
            self.buffer_ser2net = bytearray()
            if self.socket is not None:
                self.socket.close()
                self.socket = None
                if self.log is not None:
                    self.log.warning('{}: Disconnected'.format(self.device))

def test():
    if False:
        while True:
            i = 10
    service = ZeroconfService(name='TestService', port=3000)
    service.publish()
    input('Press the ENTER key to unpublish the service ')
    service.unpublish()
if __name__ == '__main__':
    import logging
    import argparse
    VERBOSTIY = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    parser = argparse.ArgumentParser(usage='%(prog)s [options]\n\nAnnounce the existence of devices using zeroconf and provide\na TCP/IP <-> serial port gateway (implements RFC 2217).\n\nIf running as daemon, write to syslog. Otherwise write to stdout.\n', epilog='NOTE: no security measures are implemented. Anyone can remotely connect\nto this service over the network.\n\nOnly one connection at once, per port, is supported. When the connection is\nterminated, it waits for the next connect.\n')
    group = parser.add_argument_group('serial port settings')
    group.add_argument('--ports-regex', help='specify a regex to search against the serial devices and their descriptions (default: %(default)s)', default='/dev/ttyUSB[0-9]+', metavar='REGEX')
    group = parser.add_argument_group('network settings')
    group.add_argument('--tcp-port', dest='base_port', help='specify lowest TCP port number (default: %(default)s)', default=7000, type=int, metavar='PORT')
    group = parser.add_argument_group('daemon')
    group.add_argument('-d', '--daemon', dest='daemonize', action='store_true', help='start as daemon', default=False)
    group.add_argument('--pidfile', help='specify a name for the PID file', default=None, metavar='FILE')
    group = parser.add_argument_group('diagnostics')
    group.add_argument('-o', '--logfile', help='write messages file instead of stdout', default=None, metavar='FILE')
    group.add_argument('-q', '--quiet', dest='verbosity', action='store_const', const=0, help='suppress most diagnostic messages', default=1)
    group.add_argument('-v', '--verbose', dest='verbosity', action='count', help='increase diagnostic messages')
    args = parser.parse_args()
    logging.basicConfig(level=VERBOSTIY[min(args.verbosity, len(VERBOSTIY) - 1)])
    log = logging.getLogger('port_publisher')
    if args.logfile is not None:

        class WriteFlushed:

            def __init__(self, fileobj):
                if False:
                    print('Hello World!')
                self.fileobj = fileobj

            def write(self, s):
                if False:
                    print('Hello World!')
                self.fileobj.write(s)
                self.fileobj.flush()

            def close(self):
                if False:
                    while True:
                        i = 10
                self.fileobj.close()
        sys.stdout = sys.stderr = WriteFlushed(open(args.logfile, 'a'))
    if args.daemonize:
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            log.critical('fork #1 failed: {} ({})\n'.format(e.errno, e.strerror))
            sys.exit(1)
        os.chdir('/')
        os.setsid()
        os.umask(0)
        try:
            pid = os.fork()
            if pid > 0:
                if args.pidfile is not None:
                    open(args.pidfile, 'w').write('{}'.format(pid))
                sys.exit(0)
        except OSError as e:
            log.critical('fork #2 failed: {} ({})\n'.format(e.errno, e.strerror))
            sys.exit(1)
        if args.logfile is None:
            import syslog
            syslog.openlog('serial port publisher')

            class WriteToSysLog:

                def __init__(self):
                    if False:
                        return 10
                    self.buffer = ''

                def write(self, s):
                    if False:
                        return 10
                    self.buffer += s
                    if '\n' in self.buffer:
                        (output, self.buffer) = self.buffer.split('\n', 1)
                        syslog.syslog(output)

                def flush(self):
                    if False:
                        return 10
                    syslog.syslog(self.buffer)
                    self.buffer = ''

                def close(self):
                    if False:
                        return 10
                    self.flush()
            sys.stdout = sys.stderr = WriteToSysLog()
    published = {}
    hostname = socket.gethostname()

    def unpublish(forwarder):
        if False:
            return 10
        'when forwarders die, we need to unregister them'
        try:
            del published[forwarder.device]
        except KeyError:
            pass
        else:
            log.info('unpublish: {}'.format(forwarder))
    alive = True
    next_check = 0
    while alive:
        try:
            now = time.time()
            if now > next_check:
                next_check = now + 5
                connected = [d for (d, p, i) in serial.tools.list_ports.grep(args.ports_regex)]
                for device in set(published).difference(connected):
                    log.info('unpublish: {}'.format(published[device]))
                    unpublish(published[device])
                for device in sorted(set(connected).difference(published)):
                    port = args.base_port
                    ports_in_use = [f.network_port for f in published.values()]
                    while port in ports_in_use:
                        port += 1
                    published[device] = Forwarder(device, '{} on {}'.format(device, hostname), port, on_close=unpublish, log=log)
                    log.warning('publish: {}'.format(published[device]))
                    published[device].open()
            read_map = {}
            write_map = {}
            error_map = {}
            for publisher in published.values():
                publisher.update_select_maps(read_map, write_map, error_map)
            (readers, writers, errors) = select.select(read_map.keys(), write_map.keys(), error_map.keys(), 5)
            for reader in readers:
                read_map[reader]()
            for writer in writers:
                write_map[writer]()
            for error in errors:
                error_map[error]()
        except KeyboardInterrupt:
            alive = False
            sys.stdout.write('\n')
        except SystemExit:
            raise
        except:
            traceback.print_exc()