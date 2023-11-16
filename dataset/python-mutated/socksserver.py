from __future__ import division
from __future__ import print_function
import socketserver
import socket
import time
import logging
from queue import Queue
from struct import unpack, pack
from threading import Timer, Thread
from impacket import LOG
from impacket.dcerpc.v5.enum import Enum
from impacket.structure import Structure
KEEP_ALIVE_TIMER = 30.0

class enumItems(Enum):
    NO_AUTHENTICATION = 0
    GSSAPI = 1
    USER_PASS = 2
    UNACCEPTABLE = 255

class replyField(Enum):
    SUCCEEDED = 0
    SOCKS_FAILURE = 1
    NOT_ALLOWED = 2
    NETWORK_UNREACHABLE = 3
    HOST_UNREACHABLE = 4
    CONNECTION_REFUSED = 5
    TTL_EXPIRED = 6
    COMMAND_NOT_SUPPORTED = 7
    ADDRESS_NOT_SUPPORTED = 8

class ATYP(Enum):
    IPv4 = 1
    DOMAINNAME = 3
    IPv6 = 4

class SOCKS5_GREETINGS(Structure):
    structure = (('VER', 'B=5'), ('METHODS', 'B*B'))

class SOCKS5_GREETINGS_BACK(Structure):
    structure = (('VER', 'B=5'), ('METHODS', 'B=0'))

class SOCKS5_REQUEST(Structure):
    structure = (('VER', 'B=5'), ('CMD', 'B=0'), ('RSV', 'B=0'), ('ATYP', 'B=0'), ('PAYLOAD', ':'))

class SOCKS5_REPLY(Structure):
    structure = (('VER', 'B=5'), ('REP', 'B=5'), ('RSV', 'B=0'), ('ATYP', 'B=1'), ('PAYLOAD', ':="AAAAA"'))

class SOCKS4_REQUEST(Structure):
    structure = (('VER', 'B=4'), ('CMD', 'B=0'), ('PORT', '>H=0'), ('ADDR', '4s="'), ('PAYLOAD', ':'))

class SOCKS4_REPLY(Structure):
    structure = (('VER', 'B=0'), ('REP', 'B=0x5A'), ('RSV', '<H=0'), ('RSV', '<L=0'))
activeConnections = Queue()

class RepeatedTimer(object):

    def __init__(self, interval, function, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        if False:
            i = 10
            return i + 15
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if False:
            i = 10
            return i + 15
        if not self.is_running:
            self.next_call += self.interval
            self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._timer.cancel()
        self.is_running = False

class SocksRelay:
    PLUGIN_NAME = 'Base Plugin'
    PLUGIN_SCHEME = ''

    def __init__(self, targetHost, targetPort, socksSocket, activeRelays):
        if False:
            while True:
                i = 10
        self.targetHost = targetHost
        self.targetPort = targetPort
        self.socksSocket = socksSocket
        self.sessionData = activeRelays['data']
        self.username = None
        self.clientConnection = None
        self.activeRelays = activeRelays

    def initConnection(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('Virtual Function')

    def skipAuthentication(self):
        if False:
            print('Hello World!')
        raise RuntimeError('Virtual Function')

    def tunnelConnection(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('Virtual Function')

    @staticmethod
    def getProtocolPort(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('Virtual Function')

def keepAliveTimer(server):
    if False:
        while True:
            i = 10
    LOG.debug('KeepAlive Timer reached. Updating connections')
    for target in list(server.activeRelays.keys()):
        for port in list(server.activeRelays[target].keys()):
            for user in list(server.activeRelays[target][port].keys()):
                if user != 'data' and user != 'scheme':
                    if server.activeRelays[target][port][user]['inUse'] is False:
                        LOG.debug('Calling keepAlive() for %s@%s:%s' % (user, target, port))
                        try:
                            server.activeRelays[target][port][user]['protocolClient'].keepAlive()
                        except Exception as e:
                            LOG.debug('Exception:', exc_info=True)
                            LOG.debug('SOCKS: %s' % str(e))
                            if str(e).find('Broken pipe') >= 0 or str(e).find('reset by peer') >= 0 or str(e).find('Invalid argument') >= 0 or (str(e).find('Server not connected') >= 0):
                                del server.activeRelays[target][port][user]
                                if len(list(server.activeRelays[target][port].keys())) == 1:
                                    del server.activeRelays[target][port]
                                LOG.debug('Removing active relay for %s@%s:%s' % (user, target, port))
                    else:
                        LOG.debug("Skipping %s@%s:%s since it's being used at the moment" % (user, target, port))

def activeConnectionsWatcher(server):
    if False:
        while True:
            i = 10
    while True:
        (target, port, scheme, userName, client, data) = activeConnections.get()
        if (target in server.activeRelays) is not True:
            server.activeRelays[target] = {}
        if (port in server.activeRelays[target]) is not True:
            server.activeRelays[target][port] = {}
        if (userName in server.activeRelays[target][port]) is not True:
            LOG.info('SOCKS: Adding %s@%s(%s) to active SOCKS connection. Enjoy' % (userName, target, port))
            server.activeRelays[target][port][userName] = {}
            server.activeRelays[target][port][userName]['protocolClient'] = client
            server.activeRelays[target][port][userName]['inUse'] = False
            server.activeRelays[target][port][userName]['data'] = data
            server.activeRelays[target][port]['data'] = data
            server.activeRelays[target][port]['scheme'] = scheme
            server.activeRelays[target][port][userName]['isAdmin'] = 'N/A'
            try:
                LOG.debug('Checking admin status for user %s' % str(userName))
                isAdmin = client.isAdmin()
                server.activeRelays[target][port][userName]['isAdmin'] = isAdmin
            except Exception as e:
                server.activeRelays[target][port][userName]['isAdmin'] = 'N/A'
                pass
            LOG.debug('isAdmin returned: %s' % server.activeRelays[target][port][userName]['isAdmin'])
        else:
            LOG.info('Relay connection for %s at %s(%d) already exists. Discarding' % (userName, target, port))
            client.killConnection()

def webService(server):
    if False:
        return 10
    from flask import Flask, jsonify
    app = Flask(__name__)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route('/')
    def index():
        if False:
            for i in range(10):
                print('nop')
        print(server.activeRelays)
        return 'Relays available: %s!' % len(server.activeRelays)

    @app.route('/ntlmrelayx/api/v1.0/relays', methods=['GET'])
    def get_relays():
        if False:
            for i in range(10):
                print('nop')
        relays = []
        for target in server.activeRelays:
            for port in server.activeRelays[target]:
                for user in server.activeRelays[target][port]:
                    if user != 'data' and user != 'scheme':
                        protocol = server.activeRelays[target][port]['scheme']
                        isAdmin = server.activeRelays[target][port][user]['isAdmin']
                        relays.append([protocol, target, user, isAdmin, str(port)])
        return jsonify(relays)

    @app.route('/ntlmrelayx/api/v1.0/relays', methods=['GET'])
    def get_info(relay):
        if False:
            for i in range(10):
                print('nop')
        pass
    app.run(host='0.0.0.0', port=9090)

class SocksRequestHandler(socketserver.BaseRequestHandler):

    def __init__(self, request, client_address, server):
        if False:
            for i in range(10):
                print('nop')
        self.__socksServer = server
        (self.__ip, self.__port) = client_address
        self.__connSocket = request
        self.__socksVersion = 5
        self.targetHost = None
        self.targetPort = None
        self.__NBSession = None
        socketserver.BaseRequestHandler.__init__(self, request, client_address, server)

    def sendReplyError(self, error=replyField.CONNECTION_REFUSED):
        if False:
            while True:
                i = 10
        if self.__socksVersion == 5:
            reply = SOCKS5_REPLY()
            reply['REP'] = error.value
        else:
            reply = SOCKS4_REPLY()
            if error.value != 0:
                reply['REP'] = 91
        return self.__connSocket.sendall(reply.getData())

    def handle(self):
        if False:
            for i in range(10):
                print('nop')
        LOG.debug('SOCKS: New Connection from %s(%s)' % (self.__ip, self.__port))
        data = self.__connSocket.recv(8192)
        grettings = SOCKS5_GREETINGS_BACK(data)
        self.__socksVersion = grettings['VER']
        if self.__socksVersion == 5:
            self.__connSocket.sendall(SOCKS5_GREETINGS_BACK().getData())
            data = self.__connSocket.recv(8192)
            request = SOCKS5_REQUEST(data)
        else:
            request = SOCKS4_REQUEST(data)
        if self.__socksVersion == 5:
            if request['ATYP'] == ATYP.IPv4.value:
                self.targetHost = socket.inet_ntoa(request['PAYLOAD'][:4])
                self.targetPort = unpack('>H', request['PAYLOAD'][4:])[0]
            elif request['ATYP'] == ATYP.DOMAINNAME.value:
                hostLength = unpack('!B', request['PAYLOAD'][0])[0]
                self.targetHost = request['PAYLOAD'][1:hostLength + 1]
                self.targetPort = unpack('>H', request['PAYLOAD'][hostLength + 1:])[0]
            else:
                LOG.error('No support for IPv6 yet!')
        else:
            self.targetPort = request['PORT']
            if request['ADDR'][:3] == '\x00\x00\x00' and request['ADDR'][3] != '\x00':
                nullBytePos = request['PAYLOAD'].find('\x00')
                if nullBytePos == -1:
                    LOG.error('Error while reading SOCKS4a header!')
                else:
                    self.targetHost = request['PAYLOAD'].split('\x00', 1)[1][:-1]
            else:
                self.targetHost = socket.inet_ntoa(request['ADDR'])
        LOG.debug('SOCKS: Target is %s(%s)' % (self.targetHost, self.targetPort))
        if self.targetPort != 53:
            if self.targetHost in self.__socksServer.activeRelays:
                if (self.targetPort in self.__socksServer.activeRelays[self.targetHost]) is not True:
                    LOG.error("SOCKS: Don't have a relay for %s(%s)" % (self.targetHost, self.targetPort))
                    self.sendReplyError(replyField.CONNECTION_REFUSED)
                    return
            else:
                LOG.error("SOCKS: Don't have a relay for %s(%s)" % (self.targetHost, self.targetPort))
                self.sendReplyError(replyField.CONNECTION_REFUSED)
                return
        if self.targetPort == 53:
            s = socket.socket()
            try:
                LOG.debug('SOCKS: Connecting to %s(%s)' % (self.targetHost, self.targetPort))
                s.connect((self.targetHost, self.targetPort))
            except Exception as e:
                LOG.debug('Exception:', exc_info=True)
                LOG.error('SOCKS: %s' % str(e))
                self.sendReplyError(replyField.CONNECTION_REFUSED)
                return
            if self.__socksVersion == 5:
                reply = SOCKS5_REPLY()
                reply['REP'] = replyField.SUCCEEDED.value
                (addr, port) = s.getsockname()
                reply['PAYLOAD'] = socket.inet_aton(addr) + pack('>H', port)
            else:
                reply = SOCKS4_REPLY()
            self.__connSocket.sendall(reply.getData())
            while True:
                try:
                    data = self.__connSocket.recv(8192)
                    if data == b'':
                        break
                    s.sendall(data)
                    data = s.recv(8192)
                    self.__connSocket.sendall(data)
                except Exception as e:
                    LOG.debug('Exception:', exc_info=True)
                    LOG.error('SOCKS: %s', str(e))
        scheme = None
        if self.targetHost in self.__socksServer.activeRelays:
            if self.targetPort in self.__socksServer.activeRelays[self.targetHost]:
                scheme = self.__socksServer.activeRelays[self.targetHost][self.targetPort]['scheme']
        if scheme is not None:
            LOG.debug('Handler for port %s found %s' % (self.targetPort, self.__socksServer.socksPlugins[scheme]))
            relay = self.__socksServer.socksPlugins[scheme](self.targetHost, self.targetPort, self.__connSocket, self.__socksServer.activeRelays[self.targetHost][self.targetPort])
            try:
                relay.initConnection()
                if self.__socksVersion == 5:
                    reply = SOCKS5_REPLY()
                    reply['REP'] = replyField.SUCCEEDED.value
                    (addr, port) = self.__connSocket.getsockname()
                    reply['PAYLOAD'] = socket.inet_aton(addr) + pack('>H', port)
                else:
                    reply = SOCKS4_REPLY()
                self.__connSocket.sendall(reply.getData())
                if relay.skipAuthentication() is not True:
                    self.__connSocket.close()
                    return
                self.__socksServer.activeRelays[self.targetHost][self.targetPort][relay.username]['inUse'] = True
                relay.tunnelConnection()
            except Exception as e:
                LOG.debug('Exception:', exc_info=True)
                LOG.debug('SOCKS: %s' % str(e))
                if str(e).find('Broken pipe') >= 0 or str(e).find('reset by peer') >= 0 or str(e).find('Invalid argument') >= 0:
                    del self.__socksServer.activeRelays[self.targetHost][self.targetPort][relay.username]
                    if len(list(self.__socksServer.activeRelays[self.targetHost][self.targetPort].keys())) == 1:
                        del self.__socksServer.activeRelays[self.targetHost][self.targetPort]
                    LOG.debug('Removing active relay for %s@%s:%s' % (relay.username, self.targetHost, self.targetPort))
                    self.sendReplyError(replyField.CONNECTION_REFUSED)
                    return
                pass
            if relay.username is not None:
                self.__socksServer.activeRelays[self.targetHost][self.targetPort][relay.username]['inUse'] = False
        else:
            LOG.error("SOCKS: I don't have a handler for this port")
        LOG.debug('SOCKS: Shutting down connection')
        try:
            self.sendReplyError(replyField.CONNECTION_REFUSED)
        except Exception as e:
            LOG.debug('SOCKS END: %s' % str(e))

class SOCKS(socketserver.ThreadingMixIn, socketserver.TCPServer):

    def __init__(self, server_address=('0.0.0.0', 1080), handler_class=SocksRequestHandler):
        if False:
            for i in range(10):
                print('nop')
        LOG.info('SOCKS proxy started. Listening at port %d', server_address[1])
        self.activeRelays = {}
        self.socksPlugins = {}
        self.restAPI = None
        self.activeConnectionsWatcher = None
        self.supportedSchemes = []
        socketserver.TCPServer.allow_reuse_address = True
        socketserver.TCPServer.__init__(self, server_address, handler_class)
        from impacket.examples.ntlmrelayx.servers.socksplugins import SOCKS_RELAYS
        for relay in SOCKS_RELAYS:
            LOG.info('%s loaded..' % relay.PLUGIN_NAME)
            self.socksPlugins[relay.PLUGIN_SCHEME] = relay
            self.supportedSchemes.append(relay.PLUGIN_SCHEME)
        self.__timer = RepeatedTimer(KEEP_ALIVE_TIMER, keepAliveTimer, self)
        self.restAPI = Thread(target=webService, args=(self,))
        self.restAPI.daemon = True
        self.restAPI.start()
        self.activeConnectionsWatcher = Thread(target=activeConnectionsWatcher, args=(self,))
        self.activeConnectionsWatcher.daemon = True
        self.activeConnectionsWatcher.start()

    def shutdown(self):
        if False:
            print('Hello World!')
        self.__timer.stop()
        del self.restAPI
        del self.activeConnectionsWatcher
        return socketserver.TCPServer.shutdown(self)
if __name__ == '__main__':
    from impacket.examples import logger
    logger.init()
    s = SOCKS()
    s.serve_forever()