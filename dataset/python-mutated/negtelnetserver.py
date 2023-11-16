""" A telnet server which negotiates"""
from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import sys
import logging
try:
    import SocketServer as socketserver
except ImportError:
    import socketserver
log = logging.getLogger(__name__)
HOST = 'localhost'
IDENT = 'NTEL'
VERIFIED_REQ = 'verifiedserver'
VERIFIED_RSP = 'WE ROOLZ: {pid}'

def telnetserver(options):
    if False:
        while True:
            i = 10
    '\n    Starts up a TCP server with a telnet handler and serves DICT requests\n    forever.\n    '
    if options.pidfile:
        pid = os.getpid()
        with open(options.pidfile, 'w') as f:
            f.write(str(pid))
    local_bind = (HOST, options.port)
    log.info('Listening on %s', local_bind)
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(local_bind, NegotiatingTelnetHandler)
    server.serve_forever()
    return ScriptRC.SUCCESS

class NegotiatingTelnetHandler(socketserver.BaseRequestHandler):
    """Handler class for Telnet connections.

    """

    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Negotiates options before reading data.\n        '
        neg = Negotiator(self.request)
        try:
            neg.send_do('NEW_ENVIRON')
            neg.send_will('NEW_ENVIRON')
            neg.send_dont('NAWS')
            neg.send_wont('NAWS')
            data = neg.recv(1024)
            log.debug('Incoming data: %r', data)
            if VERIFIED_REQ.encode('ascii') in data:
                log.debug('Received verification request from test framework')
                response = VERIFIED_RSP.format(pid=os.getpid())
                response_data = response.encode('ascii')
            else:
                log.debug('Received normal request - echoing back')
                response_data = data.strip()
            if response_data:
                log.debug('Sending %r', response_data)
                self.request.sendall(response_data)
        except IOError:
            log.exception('IOError hit during request')

class Negotiator(object):
    NO_NEG = 0
    START_NEG = 1
    WILL = 2
    WONT = 3
    DO = 4
    DONT = 5

    def __init__(self, tcp):
        if False:
            i = 10
            return i + 15
        self.tcp = tcp
        self.state = self.NO_NEG

    def recv(self, bytes):
        if False:
            i = 10
            return i + 15
        '\n        Read bytes from TCP, handling negotiation sequences\n\n        :param bytes: Number of bytes to read\n        :return: a buffer of bytes\n        '
        buffer = bytearray()
        while len(buffer) == 0:
            data = self.tcp.recv(bytes)
            if not data:
                break
            for byte_int in bytearray(data):
                if self.state == self.NO_NEG:
                    self.no_neg(byte_int, buffer)
                elif self.state == self.START_NEG:
                    self.start_neg(byte_int)
                elif self.state in [self.WILL, self.WONT, self.DO, self.DONT]:
                    self.handle_option(byte_int)
                else:
                    log.error('Unexpected byte %s in state %s', byte_int, self.state)
                    self.state = self.NO_NEG
        return buffer

    def no_neg(self, byte_int, buffer):
        if False:
            return 10
        if byte_int == NegTokens.IAC:
            log.debug('Starting negotiation (IAC)')
            self.state = self.START_NEG
        else:
            buffer.append(byte_int)

    def start_neg(self, byte_int):
        if False:
            i = 10
            return i + 15
        log.debug('In negotiation (%s)', NegTokens.from_val(byte_int))
        if byte_int == NegTokens.WILL:
            log.debug('Client is willing')
            self.state = self.WILL
        elif byte_int == NegTokens.WONT:
            log.debug('Client is unwilling')
            self.state = self.WONT
        elif byte_int == NegTokens.DO:
            log.debug('Client can do')
            self.state = self.DO
        elif byte_int == NegTokens.DONT:
            log.debug("Client can't do")
            self.state = self.DONT
        else:
            log.error('Unexpected byte %s in state %s', byte_int, self.state)
            self.state = self.NO_NEG

    def handle_option(self, byte_int):
        if False:
            for i in range(10):
                print('nop')
        if byte_int in [NegOptions.BINARY, NegOptions.CHARSET, NegOptions.SUPPRESS_GO_AHEAD, NegOptions.NAWS, NegOptions.NEW_ENVIRON]:
            log.debug('Option: %s', NegOptions.from_val(byte_int))
            self.state = self.NO_NEG
        else:
            log.error('Unexpected byte %s in state %s', byte_int, self.state)
            self.state = self.NO_NEG

    def send_message(self, message_ints):
        if False:
            return 10
        self.tcp.sendall(bytearray(message_ints))

    def send_iac(self, arr):
        if False:
            return 10
        message = [NegTokens.IAC]
        message.extend(arr)
        self.send_message(message)

    def send_do(self, option_str):
        if False:
            for i in range(10):
                print('nop')
        log.debug('Sending DO %s', option_str)
        self.send_iac([NegTokens.DO, NegOptions.to_val(option_str)])

    def send_dont(self, option_str):
        if False:
            i = 10
            return i + 15
        log.debug('Sending DONT %s', option_str)
        self.send_iac([NegTokens.DONT, NegOptions.to_val(option_str)])

    def send_will(self, option_str):
        if False:
            for i in range(10):
                print('nop')
        log.debug('Sending WILL %s', option_str)
        self.send_iac([NegTokens.WILL, NegOptions.to_val(option_str)])

    def send_wont(self, option_str):
        if False:
            for i in range(10):
                print('nop')
        log.debug('Sending WONT %s', option_str)
        self.send_iac([NegTokens.WONT, NegOptions.to_val(option_str)])

class NegBase(object):

    @classmethod
    def to_val(cls, name):
        if False:
            print('Hello World!')
        return getattr(cls, name)

    @classmethod
    def from_val(cls, val):
        if False:
            for i in range(10):
                print('nop')
        for k in cls.__dict__.keys():
            if getattr(cls, k) == val:
                return k
        return '<unknown>'

class NegTokens(NegBase):
    IAC = 255
    WILL = 251
    WONT = 252
    DO = 253
    DONT = 254
    SB = 250
    SE = 240

class NegOptions(NegBase):
    BINARY = 0
    SUPPRESS_GO_AHEAD = 3
    NAWS = 31
    NEW_ENVIRON = 39
    CHARSET = 42

def get_options():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', action='store', default=9019, type=int, help='port to listen on')
    parser.add_argument('--verbose', action='store', type=int, default=0, help='verbose output')
    parser.add_argument('--pidfile', action='store', help='file name for the PID')
    parser.add_argument('--logfile', action='store', help='file name for the log')
    parser.add_argument('--srcdir', action='store', help='test directory')
    parser.add_argument('--id', action='store', help='server ID')
    parser.add_argument('--ipv4', action='store_true', default=0, help='IPv4 flag')
    return parser.parse_args()

def setup_logging(options):
    if False:
        i = 10
        return i + 15
    '\n    Set up logging from the command line options\n    '
    root_logger = logging.getLogger()
    add_stdout = False
    formatter = logging.Formatter('%(asctime)s %(levelname)-5.5s [{ident}] %(message)s'.format(ident=IDENT))
    if options.logfile:
        handler = logging.FileHandler(options.logfile, mode='w')
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG)
        root_logger.addHandler(handler)
    else:
        add_stdout = True
    if options.verbose:
        root_logger.setLevel(logging.DEBUG)
        add_stdout = True
    else:
        root_logger.setLevel(logging.INFO)
    if add_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(stdout_handler)

class ScriptRC(object):
    """Enum for script return codes"""
    SUCCESS = 0
    FAILURE = 1
    EXCEPTION = 2

class ScriptException(Exception):
    pass
if __name__ == '__main__':
    options = get_options()
    setup_logging(options)
    try:
        rc = telnetserver(options)
    except Exception as e:
        log.exception(e)
        rc = ScriptRC.EXCEPTION
    log.info('Returning %d', rc)
    sys.exit(rc)