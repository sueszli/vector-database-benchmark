import sys
import socket
import serial
import serial.threaded
import time

class SerialToNet(serial.threaded.Protocol):
    """serial->socket"""

    def __init__(self):
        if False:
            return 10
        self.socket = None

    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def data_received(self, data):
        if False:
            return 10
        if self.socket is not None:
            self.socket.sendall(data)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simple Serial to Network (TCP/IP) redirector.', epilog='NOTE: no security measures are implemented. Anyone can remotely connect\nto this service over the network.\n\nOnly one connection at once is supported. When the connection is terminated\nit waits for the next connect.\n')
    parser.add_argument('SERIALPORT', help='serial port name')
    parser.add_argument('BAUDRATE', type=int, nargs='?', help='set baud rate, default: %(default)s', default=9600)
    parser.add_argument('-q', '--quiet', action='store_true', help='suppress non error messages', default=False)
    parser.add_argument('--develop', action='store_true', help='Development mode, prints Python internals on errors', default=False)
    group = parser.add_argument_group('serial port')
    group.add_argument('--bytesize', choices=[5, 6, 7, 8], type=int, help='set bytesize, one of {5 6 7 8}, default: 8', default=8)
    group.add_argument('--parity', choices=['N', 'E', 'O', 'S', 'M'], type=lambda c: c.upper(), help='set parity, one of {N E O S M}, default: N', default='N')
    group.add_argument('--stopbits', choices=[1, 1.5, 2], type=float, help='set stopbits, one of {1 1.5 2}, default: 1', default=1)
    group.add_argument('--rtscts', action='store_true', help='enable RTS/CTS flow control (default off)', default=False)
    group.add_argument('--xonxoff', action='store_true', help='enable software flow control (default off)', default=False)
    group.add_argument('--rts', type=int, help='set initial RTS line state (possible values: 0, 1)', default=None)
    group.add_argument('--dtr', type=int, help='set initial DTR line state (possible values: 0, 1)', default=None)
    group = parser.add_argument_group('network settings')
    exclusive_group = group.add_mutually_exclusive_group()
    exclusive_group.add_argument('-P', '--localport', type=int, help='local TCP port', default=7777)
    exclusive_group.add_argument('-c', '--client', metavar='HOST:PORT', help='make the connection as a client, instead of running a server', default=False)
    args = parser.parse_args()
    ser = serial.serial_for_url(args.SERIALPORT, do_not_open=True)
    ser.baudrate = args.BAUDRATE
    ser.bytesize = args.bytesize
    ser.parity = args.parity
    ser.stopbits = args.stopbits
    ser.rtscts = args.rtscts
    ser.xonxoff = args.xonxoff
    if args.rts is not None:
        ser.rts = args.rts
    if args.dtr is not None:
        ser.dtr = args.dtr
    if not args.quiet:
        sys.stderr.write('--- TCP/IP to Serial redirect on {p.name}  {p.baudrate},{p.bytesize},{p.parity},{p.stopbits} ---\n--- type Ctrl-C / BREAK to quit\n'.format(p=ser))
    try:
        ser.open()
    except serial.SerialException as e:
        sys.stderr.write('Could not open serial port {}: {}\n'.format(ser.name, e))
        sys.exit(1)
    ser_to_net = SerialToNet()
    serial_worker = serial.threaded.ReaderThread(ser, ser_to_net)
    serial_worker.start()
    if not args.client:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('', args.localport))
        srv.listen(1)
    try:
        intentional_exit = False
        while True:
            if args.client:
                (host, port) = args.client.split(':')
                sys.stderr.write('Opening connection to {}:{}...\n'.format(host, port))
                client_socket = socket.socket()
                try:
                    client_socket.connect((host, int(port)))
                except socket.error as msg:
                    sys.stderr.write('WARNING: {}\n'.format(msg))
                    time.sleep(5)
                    continue
                sys.stderr.write('Connected\n')
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            else:
                sys.stderr.write('Waiting for connection on {}...\n'.format(args.localport))
                (client_socket, addr) = srv.accept()
                sys.stderr.write('Connected by {}\n'.format(addr))
                try:
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 1)
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 1)
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                except AttributeError:
                    pass
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                ser_to_net.socket = client_socket
                while True:
                    try:
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        ser.write(data)
                    except socket.error as msg:
                        if args.develop:
                            raise
                        sys.stderr.write('ERROR: {}\n'.format(msg))
                        break
            except KeyboardInterrupt:
                intentional_exit = True
                raise
            except socket.error as msg:
                if args.develop:
                    raise
                sys.stderr.write('ERROR: {}\n'.format(msg))
            finally:
                ser_to_net.socket = None
                sys.stderr.write('Disconnected\n')
                client_socket.close()
                if args.client and (not intentional_exit):
                    time.sleep(5)
    except KeyboardInterrupt:
        pass
    sys.stderr.write('\n--- exit ---\n')
    serial_worker.stop()