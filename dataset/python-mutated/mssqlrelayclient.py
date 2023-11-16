import random
import string
from struct import unpack
from impacket import LOG
from impacket.examples.ntlmrelayx.clients import ProtocolClient
from impacket.tds import MSSQL, DummyPrint, TDS_ENCRYPT_REQ, TDS_ENCRYPT_OFF, TDS_PRE_LOGIN, TDS_LOGIN, TDS_INIT_LANG_FATAL, TDS_ODBC_ON, TDS_INTEGRATED_SECURITY_ON, TDS_LOGIN7, TDS_SSPI, TDS_LOGINACK_TOKEN
from impacket.ntlm import NTLMAuthChallenge
from impacket.nt_errors import STATUS_SUCCESS, STATUS_ACCESS_DENIED
from impacket.spnego import SPNEGO_NegTokenResp
try:
    from OpenSSL import SSL
except Exception:
    LOG.critical("pyOpenSSL is not installed, can't continue")
PROTOCOL_CLIENT_CLASS = 'MSSQLRelayClient'

class MYMSSQL(MSSQL):

    def __init__(self, address, port=1433, rowsPrinter=DummyPrint()):
        if False:
            print('Hello World!')
        MSSQL.__init__(self, address, port, rowsPrinter)
        self.resp = None
        self.sessionData = {}

    def initConnection(self):
        if False:
            print('Hello World!')
        self.connect()
        resp = self.preLogin()
        if resp['Encryption'] == TDS_ENCRYPT_REQ or resp['Encryption'] == TDS_ENCRYPT_OFF:
            LOG.debug('Encryption required, switching to TLS')
            ctx = SSL.Context(SSL.TLS_METHOD)
            ctx.set_cipher_list('ALL:@SECLEVEL=0'.encode('utf-8'))
            tls = SSL.Connection(ctx, None)
            tls.set_connect_state()
            while True:
                try:
                    tls.do_handshake()
                except SSL.WantReadError:
                    data = tls.bio_read(4096)
                    self.sendTDS(TDS_PRE_LOGIN, data, 0)
                    tds = self.recvTDS()
                    tls.bio_write(tds['Data'])
                else:
                    break
            self.packetSize = 16 * 1024 - 1
            self.tlsSocket = tls
        self.resp = resp
        return True

    def sendNegotiate(self, negotiateMessage):
        if False:
            for i in range(10):
                print('nop')
        login = TDS_LOGIN()
        login['HostName'] = ''.join([random.choice(string.ascii_letters) for _ in range(8)]).encode('utf-16le')
        login['AppName'] = ''.join([random.choice(string.ascii_letters) for _ in range(8)]).encode('utf-16le')
        login['ServerName'] = self.server.encode('utf-16le')
        login['CltIntName'] = login['AppName']
        login['ClientPID'] = random.randint(0, 1024)
        login['PacketSize'] = self.packetSize
        login['OptionFlags2'] = TDS_INIT_LANG_FATAL | TDS_ODBC_ON | TDS_INTEGRATED_SECURITY_ON
        login['SSPI'] = negotiateMessage
        login['Length'] = len(login.getData())
        self.sendTDS(TDS_LOGIN7, login.getData())
        if self.resp['Encryption'] == TDS_ENCRYPT_OFF:
            self.tlsSocket = None
        tds = self.recvTDS()
        self.sessionData['NTLM_CHALLENGE'] = tds
        challenge = NTLMAuthChallenge()
        challenge.fromString(tds['Data'][3:])
        return challenge

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            while True:
                i = 10
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken2 = SPNEGO_NegTokenResp(authenticateMessageBlob)
            token = respToken2['ResponseToken']
        else:
            token = authenticateMessageBlob
        self.sendTDS(TDS_SSPI, token)
        tds = self.recvTDS()
        self.replies = self.parseReply(tds['Data'])
        if TDS_LOGINACK_TOKEN in self.replies:
            self.sessionData['AUTH_ANSWER'] = tds
            return (None, STATUS_SUCCESS)
        else:
            self.printReplies()
            return (None, STATUS_ACCESS_DENIED)

    def close(self):
        if False:
            i = 10
            return i + 15
        return self.disconnect()

class MSSQLRelayClient(ProtocolClient):
    PLUGIN_NAME = 'MSSQL'

    def __init__(self, serverConfig, targetHost, targetPort=1433, extendedSecurity=True):
        if False:
            return 10
        ProtocolClient.__init__(self, serverConfig, targetHost, targetPort, extendedSecurity)
        self.extendedSecurity = extendedSecurity
        self.domainIp = None
        self.machineAccount = None
        self.machineHashes = None

    def initConnection(self):
        if False:
            i = 10
            return i + 15
        self.session = MYMSSQL(self.targetHost, self.targetPort)
        self.session.initConnection()
        return True

    def keepAlive(self):
        if False:
            i = 10
            return i + 15
        pass

    def killConnection(self):
        if False:
            return 10
        if self.session is not None:
            self.session.disconnect()
            self.session = None

    def sendNegotiate(self, negotiateMessage):
        if False:
            print('Hello World!')
        return self.session.sendNegotiate(negotiateMessage)

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            return 10
        self.sessionData = self.session.sessionData
        return self.session.sendAuth(authenticateMessageBlob, serverChallenge)