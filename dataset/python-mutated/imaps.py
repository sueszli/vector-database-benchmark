from impacket import LOG
from impacket.examples.ntlmrelayx.servers.socksplugins.imap import IMAPSocksRelay
from impacket.examples.ntlmrelayx.utils.ssl import SSLServerMixin
from OpenSSL import SSL
PLUGIN_CLASS = 'IMAPSSocksRelay'
EOL = '\r\n'

class IMAPSSocksRelay(SSLServerMixin, IMAPSocksRelay):
    PLUGIN_NAME = 'IMAPS Socks Plugin'
    PLUGIN_SCHEME = 'IMAPS'

    def __init__(self, targetHost, targetPort, socksSocket, activeRelays):
        if False:
            i = 10
            return i + 15
        IMAPSocksRelay.__init__(self, targetHost, targetPort, socksSocket, activeRelays)

    @staticmethod
    def getProtocolPort():
        if False:
            print('Hello World!')
        return 993

    def skipAuthentication(self):
        if False:
            i = 10
            return i + 15
        LOG.debug('Wrapping IMAP client connection in TLS/SSL')
        self.wrapClientConnection()
        try:
            if not IMAPSocksRelay.skipAuthentication(self):
                self.socksSocket.shutdown()
                return False
        except Exception as e:
            LOG.debug('IMAPS: %s' % str(e))
            return False
        self.relaySocket = self.session.sslobj
        return True

    def tunnelConnection(self):
        if False:
            i = 10
            return i + 15
        keyword = ''
        tag = ''
        while True:
            try:
                data = self.socksSocket.recv(self.packetSize)
            except SSL.ZeroReturnError:
                break
            result = self.processTunnelData(keyword, tag, data)
            if result is False:
                break
            (keyword, tag) = result
        if tag != '':
            tag = int(tag)
            if self.idleState is True:
                self.relaySocket.sendall('DONE%s' % EOL)
                self.relaySocketFile.readline()
            if self.shouldClose:
                tag += 1
                self.relaySocket.sendall('%s CLOSE%s' % (tag, EOL))
                self.relaySocketFile.readline()
            self.session.tagnum = tag + 1