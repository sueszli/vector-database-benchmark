import socket
import struct
from abc import abstractmethod, ABC
from enum import Enum
from typing import ClassVar, Optional

class ProtocolWithOpportunisticTlsEnum(str, Enum):
    """The list of plaintext protocols supported by SSLyze for opportunistic TLS upgrade (such as STARTTLS).

    This allows SSLyze to figure out how to complete an SSL/TLS handshake with the server.
    """
    SMTP = 'SMTP'
    XMPP = 'XMPP'
    XMPP_SERVER = 'XMPP_SERVER'
    FTP = 'FTP'
    POP3 = 'POP3'
    LDAP = 'LDAP'
    IMAP = 'IMAP'
    RDP = 'RDP'
    POSTGRES = 'POSTGRES'

    @classmethod
    def from_default_port(cls, port: int) -> Optional['ProtocolWithOpportunisticTlsEnum']:
        if False:
            print('Hello World!')
        'Given a port number, return the protocol that uses this port number by default.'
        try:
            return _DEFAULT_PORTS[port]
        except KeyError:
            return None
_DEFAULT_PORTS = {587: ProtocolWithOpportunisticTlsEnum.SMTP, 25: ProtocolWithOpportunisticTlsEnum.SMTP, 5222: ProtocolWithOpportunisticTlsEnum.XMPP, 5269: ProtocolWithOpportunisticTlsEnum.XMPP_SERVER, 109: ProtocolWithOpportunisticTlsEnum.POP3, 110: ProtocolWithOpportunisticTlsEnum.POP3, 143: ProtocolWithOpportunisticTlsEnum.IMAP, 220: ProtocolWithOpportunisticTlsEnum.IMAP, 21: ProtocolWithOpportunisticTlsEnum.FTP, 3268: ProtocolWithOpportunisticTlsEnum.LDAP, 389: ProtocolWithOpportunisticTlsEnum.LDAP, 3389: ProtocolWithOpportunisticTlsEnum.RDP, 5432: ProtocolWithOpportunisticTlsEnum.POSTGRES}

class OpportunisticTlsError(Exception):
    pass

class _OpportunisticTlsHelper(ABC):

    @abstractmethod
    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            return 10
        'Send the right protocol-specific requests to prepare the server for the TLS handshake.'
        pass

class _SmtpHelper(_OpportunisticTlsHelper):
    """Perform an SMTP StartTLS negotiation."""

    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            i = 10
            return i + 15
        sock.recv(2048)
        sock.send(b'EHLO sslyze.scan\r\n')
        if b'250 ' not in sock.recv(2048):
            raise OpportunisticTlsError('SMTP EHLO was rejected')
        sock.send(b'STARTTLS\r\n')
        if b'220' not in sock.recv(2048):
            raise OpportunisticTlsError('SMTP STARTTLS not supported')

class _XmppHelper(_OpportunisticTlsHelper):
    """Perform an XMPP StartTLS negotiation."""
    XMPP_OPEN_STREAM = "<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' xmlns:tls='http://www.ietf.org/rfc/rfc2595.txt' to='{xmpp_to}' xml:lang='en' version='1.0'>"
    XMPP_STARTTLS = b"<starttls xmlns='urn:ietf:params:xml:ns:xmpp-tls'/>"

    def __init__(self, xmpp_to: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._xmpp_to = xmpp_to

    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            while True:
                i = 10
        sock.send(self.XMPP_OPEN_STREAM.format(xmpp_to=self._xmpp_to).encode('utf-8'))
        server_resp = sock.recv(4096)
        if b'<stream:error>' in server_resp:
            raise OpportunisticTlsError('Error opening XMPP stream, try --xmpp_to')
        elif b'</stream:features>' not in server_resp:
            sock.recv(4096)
        sock.send(self.XMPP_STARTTLS)
        xmpp_resp = sock.recv(2048)
        if b'host-unknown' in xmpp_resp:
            raise OpportunisticTlsError('Error opening XMPP stream: server returned host-unknown error, try --xmpp_to')
        if b'proceed' not in xmpp_resp:
            raise OpportunisticTlsError('XMPP STARTTLS not supported')

class _XmppServerHelper(_XmppHelper):
    XMPP_OPEN_STREAM = "<stream:stream xmlns='jabber:server' xmlns:stream='http://etherx.jabber.org/streams' xmlns:tls='http://www.ietf.org/rfc/rfc2595.txt' to='{xmpp_to}' xml:lang='en' version='1.0'>"

class _LdapHelper(_OpportunisticTlsHelper):
    """Performs an LDAP StartTLS negotiation."""
    START_TLS_CMD = b'0\x1d\x02\x01\x01w\x18\x80\x161.3.6.1.4.1.1466.20037'
    START_TLS_OK = b'0\x0c\x02\x01\x01x\x07\n\x01\x00\x04\x00\x04'
    START_TLS_OK2 = b'Start TLS request accepted'
    START_TLS_OK_APACHEDS = b'0&\x02\x01\x01x!\n\x01\x00\x04\x00\x04\x00\x8a\x161.3.6.1.4.1.1466.20037\x8b\x00'

    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            while True:
                i = 10
        sock.send(self.START_TLS_CMD)
        data = sock.recv(2048)
        if self.START_TLS_OK not in data and self.START_TLS_OK_APACHEDS not in data and (self.START_TLS_OK2 not in data):
            raise OpportunisticTlsError(f'LDAP AUTH TLS was rejected; returned: {repr(data)}')

class _RdpHelper(_OpportunisticTlsHelper):
    """Perform an RDP StartTLS negotiation."""
    ERR_NO_STARTTLS = 'RDP AUTH TLS was rejected'
    START_TLS_CMD = b'\x03\x00\x00\x13\x0e\xe0\x00\x00\x00\x00\x00\x01\x00\x08\x00\x03\x00\x00\x00'
    START_TLS_OK = b'Start TLS request accepted.'

    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            while True:
                i = 10
        sock.send(self.START_TLS_CMD)
        data = sock.recv(4)
        if not data or len(data) != 4 or data[:2] != b'\x03\x00':
            raise OpportunisticTlsError(self.ERR_NO_STARTTLS)
        packet_len = struct.unpack('>H', data[2:])[0] - 4
        data = sock.recv(packet_len)
        if not data or len(data) != packet_len:
            raise OpportunisticTlsError(self.ERR_NO_STARTTLS)

class _GenericOpportunisticTlsHelper(_OpportunisticTlsHelper, ABC):
    ERR_NO_STARTTLS: ClassVar[str]
    START_TLS_CMD: ClassVar[bytes]
    START_TLS_OK: ClassVar[bytes]
    SHOULD_WAIT_FOR_SERVER_BANNER = True

    def prepare_socket_for_tls_handshake(self, sock: socket.socket) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.SHOULD_WAIT_FOR_SERVER_BANNER:
            sock.recv(2048)
        sock.send(self.START_TLS_CMD)
        if self.START_TLS_OK not in sock.recv(2048):
            raise OpportunisticTlsError(self.ERR_NO_STARTTLS)

class _ImapHelper(_GenericOpportunisticTlsHelper):
    ERR_NO_STARTTLS = 'IMAP START TLS was rejected'
    START_TLS_CMD = b'. STARTTLS\r\n'
    START_TLS_OK = b'. OK'

class _Pop3Helper(_GenericOpportunisticTlsHelper):
    ERR_NO_STARTTLS = 'POP START TLS was rejected'
    START_TLS_CMD = b'STLS\r\n'
    START_TLS_OK = b'+OK'

class _FtpHelper(_GenericOpportunisticTlsHelper):
    ERR_NO_STARTTLS = 'FTP AUTH TLS was rejected'
    START_TLS_CMD = b'AUTH TLS\r\n'
    START_TLS_OK = b'234'

class _PostgresHelper(_GenericOpportunisticTlsHelper):
    ERR_NO_STARTTLS = 'Postgres AUTH TLS was rejected'
    START_TLS_CMD = b'\x00\x00\x00\x08\x04\xd2\x16/'
    START_TLS_OK = b'S'
    SHOULD_WAIT_FOR_SERVER_BANNER = False
_START_TLS_HELPER_CLASSES = {ProtocolWithOpportunisticTlsEnum.SMTP: _SmtpHelper, ProtocolWithOpportunisticTlsEnum.XMPP: _XmppHelper, ProtocolWithOpportunisticTlsEnum.XMPP_SERVER: _XmppServerHelper, ProtocolWithOpportunisticTlsEnum.POP3: _Pop3Helper, ProtocolWithOpportunisticTlsEnum.IMAP: _ImapHelper, ProtocolWithOpportunisticTlsEnum.FTP: _FtpHelper, ProtocolWithOpportunisticTlsEnum.LDAP: _LdapHelper, ProtocolWithOpportunisticTlsEnum.RDP: _RdpHelper, ProtocolWithOpportunisticTlsEnum.POSTGRES: _PostgresHelper}

def get_opportunistic_tls_helper(protocol: ProtocolWithOpportunisticTlsEnum, xmpp_to_hostname: Optional[str]) -> _OpportunisticTlsHelper:
    if False:
        while True:
            i = 10
    helper_cls = _START_TLS_HELPER_CLASSES[protocol]
    if protocol not in [ProtocolWithOpportunisticTlsEnum.XMPP, ProtocolWithOpportunisticTlsEnum.XMPP_SERVER]:
        opportunistic_tls_helper = helper_cls()
    else:
        if xmpp_to_hostname is None:
            raise ValueError('Received None for xmpp_to_hostname')
        opportunistic_tls_helper = helper_cls(xmpp_to=xmpp_to_hostname)
    return opportunistic_tls_helper