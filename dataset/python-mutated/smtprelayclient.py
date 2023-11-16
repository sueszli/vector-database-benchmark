import smtplib
import base64
from struct import unpack
from impacket import LOG
from impacket.examples.ntlmrelayx.clients import ProtocolClient
from impacket.nt_errors import STATUS_SUCCESS, STATUS_ACCESS_DENIED
from impacket.ntlm import NTLMAuthChallenge
from impacket.spnego import SPNEGO_NegTokenResp
PROTOCOL_CLIENT_CLASSES = ['SMTPRelayClient']

class SMTPRelayClient(ProtocolClient):
    PLUGIN_NAME = 'SMTP'

    def __init__(self, serverConfig, target, targetPort=25, extendedSecurity=True):
        if False:
            print('Hello World!')
        ProtocolClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)

    def initConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.session = smtplib.SMTP(self.targetHost, self.targetPort)
        self.session.ehlo()
        if 'AUTH NTLM' not in self.session.ehlo_resp:
            LOG.error('SMTP server does not support NTLM authentication!')
            return False
        return True

    def sendNegotiate(self, negotiateMessage):
        if False:
            print('Hello World!')
        negotiate = base64.b64encode(negotiateMessage)
        self.session.putcmd('AUTH NTLM')
        (code, resp) = self.session.getreply()
        if code != 334:
            LOG.error('SMTP Client error, expected 334 NTLM supported, got %d %s ' % (code, resp))
            return False
        else:
            self.session.putcmd(negotiate)
        try:
            (code, serverChallengeBase64) = self.session.getreply()
            serverChallenge = base64.b64decode(serverChallengeBase64)
            challenge = NTLMAuthChallenge()
            challenge.fromString(serverChallenge)
            return challenge
        except (IndexError, KeyError, AttributeError):
            LOG.error('No NTLM challenge returned from SMTP server')
            raise

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            print('Hello World!')
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken2 = SPNEGO_NegTokenResp(authenticateMessageBlob)
            token = respToken2['ResponseToken']
        else:
            token = authenticateMessageBlob
        auth = base64.b64encode(token)
        self.session.putcmd(auth)
        (typ, data) = self.session.getreply()
        if typ == 235:
            self.session.state = 'AUTH'
            return (None, STATUS_SUCCESS)
        else:
            LOG.error('SMTP: %s' % ''.join(data))
            return (None, STATUS_ACCESS_DENIED)

    def killConnection(self):
        if False:
            return 10
        if self.session is not None:
            self.session.close()
            self.session = None

    def keepAlive(self):
        if False:
            i = 10
            return i + 15
        self.session.noop()