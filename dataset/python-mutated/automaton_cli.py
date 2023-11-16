"""
TLS client automaton. This makes for a primitive TLS stack.
Obviously you need rights for network access.

We support versions SSLv2 to TLS 1.3, along with many features.

In order to run a client to tcp/50000 with one cipher suite of your choice::

    from scapy.layers.tls import *
    ch = TLSClientHello(ciphers=<int code of the cipher suite>)
    t = TLSClientAutomaton(dport=50000, client_hello=ch)
    t.run()

You can also use it as a SuperSocket using the ``tlslink`` io::

    from scapy.layers.tls import *
    a = TLSClientAutomaton.tlslink(Raw, server="scapy.net", dport=443)
    a.send(HTTP()/HTTPRequest())
    while True:
        a.recv()

You can also use the io with a TCPSession, e.g. to get an HTTPS answer::

    from scapy.all import *
    from scapy.layers.http import *
    from scapy.layers.tls import *
    a = TLSClientAutomaton.tlslink(HTTP, server="www.google.com", dport=443)
    pkt = a.sr1(HTTP()/HTTPRequest(), session=TCPSession(app=True),
                timeout=2)
"""
import socket
import binascii
import struct
import time
from scapy.config import conf
from scapy.utils import randstring, repr_hex
from scapy.automaton import ATMT, select_objects
from scapy.error import warning
from scapy.layers.tls.automaton import _TLSAutomaton
from scapy.layers.tls.basefields import _tls_version, _tls_version_options
from scapy.layers.tls.session import tlsSession
from scapy.layers.tls.extensions import TLS_Ext_SupportedGroups, TLS_Ext_SupportedVersion_CH, TLS_Ext_SignatureAlgorithms, TLS_Ext_SupportedVersion_SH, TLS_Ext_PSKKeyExchangeModes, TLS_Ext_ServerName, ServerName
from scapy.layers.tls.handshake import TLSCertificate, TLSCertificateRequest, TLSCertificateVerify, TLSClientHello, TLSClientKeyExchange, TLSEncryptedExtensions, TLSFinished, TLSServerHello, TLSServerHelloDone, TLSServerKeyExchange, TLS13Certificate, TLS13ClientHello, TLS13ServerHello, TLS13HelloRetryRequest, TLS13CertificateRequest, _ASN1CertAndExt, TLS13KeyUpdate, TLS13NewSessionTicket
from scapy.layers.tls.handshake_sslv2 import SSLv2ClientHello, SSLv2ServerHello, SSLv2ClientMasterKey, SSLv2ServerVerify, SSLv2ClientFinished, SSLv2ServerFinished, SSLv2ClientCertificate, SSLv2RequestCertificate
from scapy.layers.tls.keyexchange_tls13 import TLS_Ext_KeyShare_CH, KeyShareEntry, TLS_Ext_KeyShare_HRR, PSKIdentity, PSKBinderEntry, TLS_Ext_PreSharedKey_CH
from scapy.layers.tls.record import TLSAlert, TLSChangeCipherSpec, TLSApplicationData
from scapy.layers.tls.crypto.suites import _tls_cipher_suites, _tls_cipher_suites_cls
from scapy.layers.tls.crypto.groups import _tls_named_groups
from scapy.layers.tls.crypto.hkdf import TLS13_HKDF
from scapy.packet import Raw
from scapy.compat import bytes_encode

class TLSClientAutomaton(_TLSAutomaton):
    """
    A simple TLS test client automaton. Try to overload some states or
    conditions and see what happens on the other side.

    Rather than with an interruption, the best way to stop this client is by
    typing 'quit'. This won't be a message sent to the server.

    :param server: the server IP or hostname. defaults to 127.0.0.1
    :param dport: the server port. defaults to 4433
    :param server_name: the SNI to use. It does not need to be set
    :param mycert:
    :param mykey: may be provided as filenames. They will be used in
        the handshake, should the server ask for client authentication.
    :param client_hello: may hold a TLSClientHello or SSLv2ClientHello to be
        sent to the server. This is particularly useful for extensions
        tweaking. If not set, a default is populated accordingly.
    :param version: is a quicker way to advertise a protocol version ("sslv2",
        "tls1", "tls12", etc.) It may be overridden by the previous
        'client_hello'.
    :param data: is a list of raw data to be sent to the server once the
        handshake has been completed. Both 'stop_server' and 'quit' will
        work this way.
    """

    def parse_args(self, server='127.0.0.1', dport=4433, server_name=None, mycert=None, mykey=None, client_hello=None, version=None, resumption_master_secret=None, session_ticket_file_in=None, session_ticket_file_out=None, psk=None, psk_mode=None, data=None, ciphersuite=None, curve=None, **kargs):
        if False:
            while True:
                i = 10
        super(TLSClientAutomaton, self).parse_args(mycert=mycert, mykey=mykey, **kargs)
        tmp = socket.getaddrinfo(server, dport)
        self.remote_family = tmp[0][0]
        self.remote_ip = tmp[0][4][0]
        self.remote_port = dport
        self.server_name = server_name
        self.local_ip = None
        self.local_port = None
        self.socket = None
        if isinstance(client_hello, (TLSClientHello, TLS13ClientHello)):
            self.client_hello = client_hello
        else:
            self.client_hello = None
        self.advertised_tls_version = None
        if version:
            v = _tls_version_options.get(version, None)
            if not v:
                self.vprint('Unrecognized TLS version option.')
            else:
                self.advertised_tls_version = v
        self.linebreak = False
        if isinstance(data, bytes):
            self.data_to_send = [data]
        elif isinstance(data, str):
            self.data_to_send = [bytes_encode(data)]
        elif isinstance(data, list):
            self.data_to_send = list((bytes_encode(d) for d in reversed(data)))
        else:
            self.data_to_send = []
        self.curve = None
        if self.advertised_tls_version == 772:
            self.ciphersuite = 4865
            if ciphersuite is not None:
                cs = int(ciphersuite, 16)
                if cs in _tls_cipher_suites.keys():
                    self.ciphersuite = cs
            if conf.crypto_valid_advanced:
                self.curve = 29
            else:
                self.curve = 23
            self.resumption_master_secret = resumption_master_secret
            self.session_ticket_file_in = session_ticket_file_in
            self.session_ticket_file_out = session_ticket_file_out
            self.tls13_psk_secret = psk
            self.tls13_psk_mode = psk_mode
            if curve is not None:
                for (group_id, ng) in _tls_named_groups.items():
                    if ng == curve:
                        if curve == 'x25519':
                            if conf.crypto_valid_advanced:
                                self.curve = group_id
                        else:
                            self.curve = group_id

    def vprint_sessioninfo(self):
        if False:
            while True:
                i = 10
        if self.verbose:
            s = self.cur_session
            v = _tls_version[s.tls_version]
            self.vprint('Version       : %s' % v)
            cs = s.wcs.ciphersuite.name
            self.vprint('Cipher suite  : %s' % cs)
            if s.tls_version >= 772:
                ms = s.tls13_master_secret
            else:
                ms = s.master_secret
            self.vprint('Master secret : %s' % repr_hex(ms))
            if s.server_certs:
                self.vprint('Server certificate chain: %r' % s.server_certs)
            if s.tls_version >= 772:
                res_secret = s.tls13_derived_secrets['resumption_secret']
                self.vprint('Resumption master secret : %s' % repr_hex(res_secret))
            self.vprint()

    @ATMT.state(initial=True)
    def INITIAL(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('Starting TLS client automaton.')
        raise self.INIT_TLS_SESSION()

    @ATMT.ioevent(INITIAL, name='tls', as_supersocket='tlslink')
    def _socket(self, fd):
        if False:
            while True:
                i = 10
        pass

    @ATMT.state()
    def INIT_TLS_SESSION(self):
        if False:
            while True:
                i = 10
        self.cur_session = tlsSession(connection_end='client')
        s = self.cur_session
        s.client_certs = self.mycert
        s.client_key = self.mykey
        v = self.advertised_tls_version
        if v:
            s.advertised_tls_version = v
        else:
            default_version = s.advertised_tls_version
            self.advertised_tls_version = default_version
        if s.advertised_tls_version >= 772:
            if self.tls13_psk_secret:
                s.tls13_psk_secret = binascii.unhexlify(self.tls13_psk_secret)
            if self.session_ticket_file_in:
                with open(self.session_ticket_file_in, 'rb') as f:
                    resumed_ciphersuite_len = struct.unpack('B', f.read(1))[0]
                    s.tls13_ticket_ciphersuite = struct.unpack('!H', f.read(resumed_ciphersuite_len))[0]
                    ticket_nonce_len = struct.unpack('B', f.read(1))[0]
                    s.client_session_nonce = f.read(ticket_nonce_len)
                    client_ticket_age_len = struct.unpack('!H', f.read(2))[0]
                    tmp = f.read(client_ticket_age_len)
                    s.client_ticket_age = struct.unpack('!I', tmp)[0]
                    client_ticket_age_add_len = struct.unpack('!H', f.read(2))[0]
                    tmp = f.read(client_ticket_age_add_len)
                    s.client_session_ticket_age_add = struct.unpack('!I', tmp)[0]
                    ticket_len = struct.unpack('!H', f.read(2))[0]
                    s.client_session_ticket = f.read(ticket_len)
                if self.resumption_master_secret:
                    if s.tls13_ticket_ciphersuite not in _tls_cipher_suites_cls:
                        warning('Unknown cipher suite %d', s.tls13_ticket_ciphersuite)
                    else:
                        cs_cls = _tls_cipher_suites_cls[s.tls13_ticket_ciphersuite]
                    hkdf = TLS13_HKDF(cs_cls.hash_alg.name.lower())
                    hash_len = hkdf.hash.digest_size
                    s.tls13_psk_secret = hkdf.expand_label(binascii.unhexlify(self.resumption_master_secret), b'resumption', s.client_session_nonce, hash_len)
        raise self.CONNECT()

    @ATMT.state()
    def CONNECT(self):
        if False:
            for i in range(10):
                print('nop')
        s = socket.socket(self.remote_family, socket.SOCK_STREAM)
        self.vprint()
        self.vprint('Trying to connect on %s:%d' % (self.remote_ip, self.remote_port))
        s.connect((self.remote_ip, self.remote_port))
        self.socket = s
        (self.local_ip, self.local_port) = self.socket.getsockname()[:2]
        self.vprint()
        if self.cur_session.advertised_tls_version in [512, 2]:
            raise self.SSLv2_PREPARE_CLIENTHELLO()
        elif self.cur_session.advertised_tls_version >= 772:
            raise self.TLS13_START()
        else:
            raise self.PREPARE_CLIENTFLIGHT1()

    @ATMT.state()
    def PREPARE_CLIENTFLIGHT1(self):
        if False:
            print('Hello World!')
        self.add_record()

    @ATMT.condition(PREPARE_CLIENTFLIGHT1)
    def should_add_ClientHello(self):
        if False:
            while True:
                i = 10
        if self.client_hello:
            p = self.client_hello
        else:
            p = TLSClientHello()
        ext = []
        if self.cur_session.advertised_tls_version == 771:
            ext += [TLS_Ext_SignatureAlgorithms(sig_algs=['sha256+rsa'])]
        if self.server_name:
            ext += TLS_Ext_ServerName(servernames=[ServerName(servername=self.server_name)])
        p.ext = ext
        self.add_msg(p)
        raise self.ADDED_CLIENTHELLO()

    @ATMT.state()
    def ADDED_CLIENTHELLO(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.condition(ADDED_CLIENTHELLO)
    def should_send_ClientFlight1(self):
        if False:
            i = 10
            return i + 15
        self.flush_records()
        raise self.SENT_CLIENTFLIGHT1()

    @ATMT.state()
    def SENT_CLIENTFLIGHT1(self):
        if False:
            return 10
        raise self.WAITING_SERVERFLIGHT1()

    @ATMT.state()
    def WAITING_SERVERFLIGHT1(self):
        if False:
            return 10
        self.get_next_msg()
        raise self.RECEIVED_SERVERFLIGHT1()

    @ATMT.state()
    def RECEIVED_SERVERFLIGHT1(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.condition(RECEIVED_SERVERFLIGHT1, prio=1)
    def should_handle_ServerHello(self):
        if False:
            i = 10
            return i + 15
        '\n        XXX We should check the ServerHello attributes for discrepancies with\n        our own ClientHello.\n        '
        self.raise_on_packet(TLSServerHello, self.HANDLED_SERVERHELLO)

    @ATMT.state()
    def HANDLED_SERVERHELLO(self):
        if False:
            return 10
        pass

    @ATMT.condition(RECEIVED_SERVERFLIGHT1, prio=2)
    def missing_ServerHello(self):
        if False:
            return 10
        raise self.MISSING_SERVERHELLO()

    @ATMT.state()
    def MISSING_SERVERHELLO(self):
        if False:
            return 10
        self.vprint('Missing TLS ServerHello message!')
        raise self.CLOSE_NOTIFY()

    @ATMT.condition(HANDLED_SERVERHELLO, prio=1)
    def should_handle_ServerCertificate(self):
        if False:
            return 10
        if not self.cur_session.prcs.key_exchange.anonymous:
            self.raise_on_packet(TLSCertificate, self.HANDLED_SERVERCERTIFICATE)
        raise self.HANDLED_SERVERCERTIFICATE()

    @ATMT.state()
    def HANDLED_SERVERCERTIFICATE(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(HANDLED_SERVERHELLO, prio=2)
    def missing_ServerCertificate(self):
        if False:
            print('Hello World!')
        raise self.MISSING_SERVERCERTIFICATE()

    @ATMT.state()
    def MISSING_SERVERCERTIFICATE(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('Missing TLS Certificate message!')
        raise self.CLOSE_NOTIFY()

    @ATMT.state()
    def HANDLED_CERTIFICATEREQUEST(self):
        if False:
            return 10
        self.vprint('Server asked for a certificate...')
        if not self.mykey or not self.mycert:
            self.vprint('No client certificate to send!')
            self.vprint('Will try and send an empty Certificate message...')

    @ATMT.condition(HANDLED_SERVERCERTIFICATE, prio=1)
    def should_handle_ServerKeyExchange_from_ServerCertificate(self):
        if False:
            return 10
        '\n        XXX We should check the ServerKeyExchange attributes for discrepancies\n        with our own ClientHello, along with the ServerHello and Certificate.\n        '
        self.raise_on_packet(TLSServerKeyExchange, self.HANDLED_SERVERKEYEXCHANGE)

    @ATMT.state(final=True)
    def MISSING_SERVERKEYEXCHANGE(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.condition(HANDLED_SERVERCERTIFICATE, prio=2)
    def missing_ServerKeyExchange(self):
        if False:
            i = 10
            return i + 15
        if not self.cur_session.prcs.key_exchange.no_ske:
            raise self.MISSING_SERVERKEYEXCHANGE()

    @ATMT.state()
    def HANDLED_SERVERKEYEXCHANGE(self):
        if False:
            while True:
                i = 10
        pass

    def should_handle_CertificateRequest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        XXX We should check the CertificateRequest attributes for discrepancies\n        with the cipher suite, etc.\n        '
        self.raise_on_packet(TLSCertificateRequest, self.HANDLED_CERTIFICATEREQUEST)

    @ATMT.condition(HANDLED_SERVERKEYEXCHANGE, prio=2)
    def should_handle_CertificateRequest_from_ServerKeyExchange(self):
        if False:
            print('Hello World!')
        self.should_handle_CertificateRequest()

    @ATMT.condition(HANDLED_SERVERCERTIFICATE, prio=3)
    def should_handle_CertificateRequest_from_ServerCertificate(self):
        if False:
            while True:
                i = 10
        self.should_handle_CertificateRequest()

    def should_handle_ServerHelloDone(self):
        if False:
            print('Hello World!')
        self.raise_on_packet(TLSServerHelloDone, self.HANDLED_SERVERHELLODONE)

    @ATMT.condition(HANDLED_SERVERKEYEXCHANGE, prio=1)
    def should_handle_ServerHelloDone_from_ServerKeyExchange(self):
        if False:
            i = 10
            return i + 15
        return self.should_handle_ServerHelloDone()

    @ATMT.condition(HANDLED_CERTIFICATEREQUEST, prio=4)
    def should_handle_ServerHelloDone_from_CertificateRequest(self):
        if False:
            while True:
                i = 10
        return self.should_handle_ServerHelloDone()

    @ATMT.condition(HANDLED_SERVERCERTIFICATE, prio=4)
    def should_handle_ServerHelloDone_from_ServerCertificate(self):
        if False:
            while True:
                i = 10
        return self.should_handle_ServerHelloDone()

    @ATMT.state()
    def HANDLED_SERVERHELLODONE(self):
        if False:
            print('Hello World!')
        raise self.PREPARE_CLIENTFLIGHT2()

    @ATMT.state()
    def PREPARE_CLIENTFLIGHT2(self):
        if False:
            while True:
                i = 10
        self.add_record()

    @ATMT.condition(PREPARE_CLIENTFLIGHT2, prio=1)
    def should_add_ClientCertificate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the server sent a CertificateRequest, we send a Certificate message.\n        If no certificate is available, an empty Certificate message is sent:\n        - this is a SHOULD in RFC 4346 (Section 7.4.6)\n        - this is a MUST in RFC 5246 (Section 7.4.6)\n\n        XXX We may want to add a complete chain.\n        '
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if TLSCertificateRequest not in hs_msg:
            return
        certs = []
        if self.mycert:
            certs = [self.mycert]
        self.add_msg(TLSCertificate(certs=certs))
        raise self.ADDED_CLIENTCERTIFICATE()

    @ATMT.state()
    def ADDED_CLIENTCERTIFICATE(self):
        if False:
            return 10
        pass

    def should_add_ClientKeyExchange(self):
        if False:
            print('Hello World!')
        self.add_msg(TLSClientKeyExchange())
        raise self.ADDED_CLIENTKEYEXCHANGE()

    @ATMT.condition(PREPARE_CLIENTFLIGHT2, prio=2)
    def should_add_ClientKeyExchange_from_ClientFlight2(self):
        if False:
            for i in range(10):
                print('nop')
        return self.should_add_ClientKeyExchange()

    @ATMT.condition(ADDED_CLIENTCERTIFICATE)
    def should_add_ClientKeyExchange_from_ClientCertificate(self):
        if False:
            return 10
        return self.should_add_ClientKeyExchange()

    @ATMT.state()
    def ADDED_CLIENTKEYEXCHANGE(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.condition(ADDED_CLIENTKEYEXCHANGE, prio=1)
    def should_add_ClientVerify(self):
        if False:
            return 10
        '\n        XXX Section 7.4.7.1 of RFC 5246 states that the CertificateVerify\n        message is only sent following a client certificate that has signing\n        capability (i.e. not those containing fixed DH params).\n        We should verify that before adding the message. We should also handle\n        the case when the Certificate message was empty.\n        '
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if TLSCertificateRequest not in hs_msg or self.mycert is None or self.mykey is None:
            return
        self.add_msg(TLSCertificateVerify())
        raise self.ADDED_CERTIFICATEVERIFY()

    @ATMT.state()
    def ADDED_CERTIFICATEVERIFY(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(ADDED_CERTIFICATEVERIFY)
    def should_add_ChangeCipherSpec_from_CertificateVerify(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_record()
        self.add_msg(TLSChangeCipherSpec())
        raise self.ADDED_CHANGECIPHERSPEC()

    @ATMT.condition(ADDED_CLIENTKEYEXCHANGE, prio=2)
    def should_add_ChangeCipherSpec_from_ClientKeyExchange(self):
        if False:
            while True:
                i = 10
        self.add_record()
        self.add_msg(TLSChangeCipherSpec())
        raise self.ADDED_CHANGECIPHERSPEC()

    @ATMT.state()
    def ADDED_CHANGECIPHERSPEC(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(ADDED_CHANGECIPHERSPEC)
    def should_add_ClientFinished(self):
        if False:
            return 10
        self.add_record()
        self.add_msg(TLSFinished())
        raise self.ADDED_CLIENTFINISHED()

    @ATMT.state()
    def ADDED_CLIENTFINISHED(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(ADDED_CLIENTFINISHED)
    def should_send_ClientFlight2(self):
        if False:
            while True:
                i = 10
        self.flush_records()
        raise self.SENT_CLIENTFLIGHT2()

    @ATMT.state()
    def SENT_CLIENTFLIGHT2(self):
        if False:
            i = 10
            return i + 15
        raise self.WAITING_SERVERFLIGHT2()

    @ATMT.state()
    def WAITING_SERVERFLIGHT2(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_next_msg()
        raise self.RECEIVED_SERVERFLIGHT2()

    @ATMT.state()
    def RECEIVED_SERVERFLIGHT2(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.condition(RECEIVED_SERVERFLIGHT2)
    def should_handle_ChangeCipherSpec(self):
        if False:
            while True:
                i = 10
        self.raise_on_packet(TLSChangeCipherSpec, self.HANDLED_CHANGECIPHERSPEC)

    @ATMT.state()
    def HANDLED_CHANGECIPHERSPEC(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(HANDLED_CHANGECIPHERSPEC)
    def should_handle_Finished(self):
        if False:
            i = 10
            return i + 15
        self.raise_on_packet(TLSFinished, self.HANDLED_SERVERFINISHED)

    @ATMT.state()
    def HANDLED_SERVERFINISHED(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('TLS handshake completed!')
        self.vprint_sessioninfo()
        self.vprint("You may send data or use 'quit'.")

    @ATMT.condition(HANDLED_SERVERFINISHED)
    def should_wait_ClientData(self):
        if False:
            i = 10
            return i + 15
        raise self.WAIT_CLIENTDATA()

    @ATMT.state()
    def WAIT_CLIENTDATA(self):
        if False:
            return 10
        pass

    @ATMT.condition(WAIT_CLIENTDATA, prio=1)
    def add_ClientData(self):
        if False:
            while True:
                i = 10
        '\n        The user may type in:\n        GET / HTTP/1.1\\r\\nHost: testserver.com\\r\\n\\r\\n\n        Special characters are handled so that it becomes a valid HTTP request.\n        '
        if not self.data_to_send:
            if self.is_atmt_socket:
                fd = select_objects([self.ioin['tls']], 0)
                if fd:
                    self.add_record()
                    self.add_msg(TLSApplicationData(data=fd[0].recv()))
                    raise self.ADDED_CLIENTDATA()
                raise self.WAITING_SERVERDATA()
            else:
                data = input().replace('\\r', '\r').replace('\\n', '\n').encode()
        else:
            data = self.data_to_send.pop()
        if data == b'quit':
            return
        elif data == b'wait':
            raise self.WAITING_SERVERDATA()
        elif data == b'key_update':
            if self.cur_session.tls_version >= 772:
                self.add_record()
                self.add_msg(TLS13KeyUpdate(request_update='update_requested'))
                raise self.ADDED_CLIENTDATA()
        if self.linebreak:
            data += b'\n'
        self.add_record()
        self.add_msg(TLSApplicationData(data=data))
        raise self.ADDED_CLIENTDATA()

    @ATMT.condition(WAIT_CLIENTDATA, prio=2)
    def no_more_ClientData(self):
        if False:
            i = 10
            return i + 15
        raise self.CLOSE_NOTIFY()

    @ATMT.state()
    def ADDED_CLIENTDATA(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(ADDED_CLIENTDATA)
    def should_send_ClientData(self):
        if False:
            return 10
        self.flush_records()
        raise self.SENT_CLIENTDATA()

    @ATMT.state()
    def SENT_CLIENTDATA(self):
        if False:
            return 10
        raise self.WAITING_SERVERDATA()

    @ATMT.state()
    def WAITING_SERVERDATA(self):
        if False:
            i = 10
            return i + 15
        self.get_next_msg(0.3, 1)
        raise self.RECEIVED_SERVERDATA()

    @ATMT.state()
    def RECEIVED_SERVERDATA(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(RECEIVED_SERVERDATA, prio=1)
    def should_handle_ServerData(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.buffer_in:
            raise self.WAIT_CLIENTDATA()
        p = self.buffer_in[0]
        if isinstance(p, TLSApplicationData):
            if self.is_atmt_socket:
                self.oi.tls.send(p.data)
            else:
                print('> Received: %r' % p.data)
        elif isinstance(p, TLSAlert):
            print('> Received: %r' % p)
            raise self.CLOSE_NOTIFY()
        elif isinstance(p, TLS13NewSessionTicket):
            print('> Received: %r ' % p)
            if self.session_ticket_file_out:
                with open(self.session_ticket_file_out, 'wb') as f:
                    f.write(struct.pack('B', 2))
                    f.write(struct.pack('!H', self.cur_session.wcs.ciphersuite.val))
                    f.write(struct.pack('B', p.noncelen))
                    f.write(p.ticket_nonce)
                    f.write(struct.pack('!H', 4))
                    f.write(struct.pack('!I', int(time.time())))
                    f.write(struct.pack('!H', 4))
                    f.write(struct.pack('!I', p.ticket_age_add))
                    f.write(struct.pack('!H', p.ticketlen))
                    f.write(self.cur_session.client_session_ticket)
        else:
            print('> Received: %r' % p)
        self.buffer_in = self.buffer_in[1:]
        raise self.HANDLED_SERVERDATA()

    @ATMT.state()
    def HANDLED_SERVERDATA(self):
        if False:
            print('Hello World!')
        raise self.WAIT_CLIENTDATA()

    @ATMT.state()
    def CLOSE_NOTIFY(self):
        if False:
            print('Hello World!')
        self.vprint()
        self.vprint('Trying to send a TLSAlert to the server...')

    @ATMT.condition(CLOSE_NOTIFY)
    def close_session(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_record()
        self.add_msg(TLSAlert(level=1, descr=0))
        try:
            self.flush_records()
        except Exception:
            self.vprint('Could not send termination Alert, maybe the server stopped?')
        raise self.FINAL()

    @ATMT.state()
    def SSLv2_PREPARE_CLIENTHELLO(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(SSLv2_PREPARE_CLIENTHELLO)
    def sslv2_should_add_ClientHello(self):
        if False:
            while True:
                i = 10
        self.add_record(is_sslv2=True)
        p = self.client_hello or SSLv2ClientHello(challenge=randstring(16))
        self.add_msg(p)
        raise self.SSLv2_ADDED_CLIENTHELLO()

    @ATMT.state()
    def SSLv2_ADDED_CLIENTHELLO(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(SSLv2_ADDED_CLIENTHELLO)
    def sslv2_should_send_ClientHello(self):
        if False:
            while True:
                i = 10
        self.flush_records()
        raise self.SSLv2_SENT_CLIENTHELLO()

    @ATMT.state()
    def SSLv2_SENT_CLIENTHELLO(self):
        if False:
            return 10
        raise self.SSLv2_WAITING_SERVERHELLO()

    @ATMT.state()
    def SSLv2_WAITING_SERVERHELLO(self):
        if False:
            print('Hello World!')
        self.get_next_msg()
        raise self.SSLv2_RECEIVED_SERVERHELLO()

    @ATMT.state()
    def SSLv2_RECEIVED_SERVERHELLO(self):
        if False:
            return 10
        pass

    @ATMT.condition(SSLv2_RECEIVED_SERVERHELLO, prio=1)
    def sslv2_should_handle_ServerHello(self):
        if False:
            i = 10
            return i + 15
        self.raise_on_packet(SSLv2ServerHello, self.SSLv2_HANDLED_SERVERHELLO)

    @ATMT.state()
    def SSLv2_HANDLED_SERVERHELLO(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(SSLv2_RECEIVED_SERVERHELLO, prio=2)
    def sslv2_missing_ServerHello(self):
        if False:
            while True:
                i = 10
        raise self.SSLv2_MISSING_SERVERHELLO()

    @ATMT.state()
    def SSLv2_MISSING_SERVERHELLO(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('Missing SSLv2 ServerHello message!')
        raise self.SSLv2_CLOSE_NOTIFY()

    @ATMT.condition(SSLv2_HANDLED_SERVERHELLO)
    def sslv2_should_add_ClientMasterKey(self):
        if False:
            while True:
                i = 10
        self.add_record(is_sslv2=True)
        self.add_msg(SSLv2ClientMasterKey())
        raise self.SSLv2_ADDED_CLIENTMASTERKEY()

    @ATMT.state()
    def SSLv2_ADDED_CLIENTMASTERKEY(self):
        if False:
            return 10
        pass

    @ATMT.condition(SSLv2_ADDED_CLIENTMASTERKEY)
    def sslv2_should_send_ClientMasterKey(self):
        if False:
            return 10
        self.flush_records()
        raise self.SSLv2_SENT_CLIENTMASTERKEY()

    @ATMT.state()
    def SSLv2_SENT_CLIENTMASTERKEY(self):
        if False:
            return 10
        raise self.SSLv2_WAITING_SERVERVERIFY()

    @ATMT.state()
    def SSLv2_WAITING_SERVERVERIFY(self):
        if False:
            while True:
                i = 10
        self.get_next_msg(0.5, 0)
        raise self.SSLv2_RECEIVED_SERVERVERIFY()

    @ATMT.state()
    def SSLv2_RECEIVED_SERVERVERIFY(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(SSLv2_RECEIVED_SERVERVERIFY, prio=1)
    def sslv2_should_handle_ServerVerify(self):
        if False:
            for i in range(10):
                print('nop')
        self.raise_on_packet(SSLv2ServerVerify, self.SSLv2_HANDLED_SERVERVERIFY, get_next_msg=False)

    @ATMT.state()
    def SSLv2_HANDLED_SERVERVERIFY(self):
        if False:
            print('Hello World!')
        pass

    def sslv2_should_add_ClientFinished(self):
        if False:
            return 10
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if SSLv2ClientFinished in hs_msg:
            return
        self.add_record(is_sslv2=True)
        self.add_msg(SSLv2ClientFinished())
        raise self.SSLv2_ADDED_CLIENTFINISHED()

    @ATMT.condition(SSLv2_HANDLED_SERVERVERIFY, prio=1)
    def sslv2_should_add_ClientFinished_from_ServerVerify(self):
        if False:
            i = 10
            return i + 15
        return self.sslv2_should_add_ClientFinished()

    @ATMT.condition(SSLv2_HANDLED_SERVERVERIFY, prio=2)
    def sslv2_should_wait_ServerFinished_from_ServerVerify(self):
        if False:
            print('Hello World!')
        raise self.SSLv2_WAITING_SERVERFINISHED()

    @ATMT.condition(SSLv2_RECEIVED_SERVERVERIFY, prio=2)
    def sslv2_should_add_ClientFinished_from_NoServerVerify(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sslv2_should_add_ClientFinished()

    @ATMT.condition(SSLv2_RECEIVED_SERVERVERIFY, prio=3)
    def sslv2_missing_ServerVerify(self):
        if False:
            while True:
                i = 10
        raise self.SSLv2_MISSING_SERVERVERIFY()

    @ATMT.state(final=True)
    def SSLv2_MISSING_SERVERVERIFY(self):
        if False:
            print('Hello World!')
        self.vprint('Missing SSLv2 ServerVerify message!')
        raise self.SSLv2_CLOSE_NOTIFY()

    @ATMT.state()
    def SSLv2_ADDED_CLIENTFINISHED(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.condition(SSLv2_ADDED_CLIENTFINISHED)
    def sslv2_should_send_ClientFinished(self):
        if False:
            i = 10
            return i + 15
        self.flush_records()
        raise self.SSLv2_SENT_CLIENTFINISHED()

    @ATMT.state()
    def SSLv2_SENT_CLIENTFINISHED(self):
        if False:
            while True:
                i = 10
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if SSLv2ServerVerify in hs_msg:
            raise self.SSLv2_WAITING_SERVERFINISHED()
        else:
            self.get_next_msg()
            raise self.SSLv2_RECEIVED_SERVERVERIFY()

    @ATMT.state()
    def SSLv2_WAITING_SERVERFINISHED(self):
        if False:
            for i in range(10):
                print('nop')
        self.get_next_msg()
        raise self.SSLv2_RECEIVED_SERVERFINISHED()

    @ATMT.state()
    def SSLv2_RECEIVED_SERVERFINISHED(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.condition(SSLv2_RECEIVED_SERVERFINISHED, prio=1)
    def sslv2_should_handle_ServerFinished(self):
        if False:
            return 10
        self.raise_on_packet(SSLv2ServerFinished, self.SSLv2_HANDLED_SERVERFINISHED)

    @ATMT.condition(SSLv2_RECEIVED_SERVERFINISHED, prio=2)
    def sslv2_should_handle_RequestCertificate(self):
        if False:
            print('Hello World!')
        self.raise_on_packet(SSLv2RequestCertificate, self.SSLv2_HANDLED_REQUESTCERTIFICATE)

    @ATMT.state()
    def SSLv2_HANDLED_REQUESTCERTIFICATE(self):
        if False:
            print('Hello World!')
        self.vprint('Server asked for a certificate...')
        if not self.mykey or not self.mycert:
            self.vprint('No client certificate to send!')
            raise self.SSLv2_CLOSE_NOTIFY()

    @ATMT.condition(SSLv2_HANDLED_REQUESTCERTIFICATE)
    def sslv2_should_add_ClientCertificate(self):
        if False:
            while True:
                i = 10
        self.add_record(is_sslv2=True)
        self.add_msg(SSLv2ClientCertificate(certdata=self.mycert))
        raise self.SSLv2_ADDED_CLIENTCERTIFICATE()

    @ATMT.state()
    def SSLv2_ADDED_CLIENTCERTIFICATE(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(SSLv2_ADDED_CLIENTCERTIFICATE)
    def sslv2_should_send_ClientCertificate(self):
        if False:
            i = 10
            return i + 15
        self.flush_records()
        raise self.SSLv2_SENT_CLIENTCERTIFICATE()

    @ATMT.state()
    def SSLv2_SENT_CLIENTCERTIFICATE(self):
        if False:
            print('Hello World!')
        raise self.SSLv2_WAITING_SERVERFINISHED()

    @ATMT.state()
    def SSLv2_HANDLED_SERVERFINISHED(self):
        if False:
            return 10
        self.vprint('SSLv2 handshake completed!')
        self.vprint_sessioninfo()
        self.vprint("You may send data or use 'quit'.")

    @ATMT.condition(SSLv2_RECEIVED_SERVERFINISHED, prio=3)
    def sslv2_missing_ServerFinished(self):
        if False:
            for i in range(10):
                print('nop')
        raise self.SSLv2_MISSING_SERVERFINISHED()

    @ATMT.state()
    def SSLv2_MISSING_SERVERFINISHED(self):
        if False:
            return 10
        self.vprint('Missing SSLv2 ServerFinished message!')
        raise self.SSLv2_CLOSE_NOTIFY()

    @ATMT.condition(SSLv2_HANDLED_SERVERFINISHED)
    def sslv2_should_wait_ClientData(self):
        if False:
            while True:
                i = 10
        raise self.SSLv2_WAITING_CLIENTDATA()

    @ATMT.state()
    def SSLv2_WAITING_CLIENTDATA(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(SSLv2_WAITING_CLIENTDATA, prio=1)
    def sslv2_add_ClientData(self):
        if False:
            return 10
        if not self.data_to_send:
            data = input().replace('\\r', '\r').replace('\\n', '\n').encode()
        else:
            data = self.data_to_send.pop()
            self.vprint('> Read from list: %s' % data)
        if data == 'quit':
            return
        if self.linebreak:
            data += '\n'
        self.add_record(is_sslv2=True)
        self.add_msg(Raw(data))
        raise self.SSLv2_ADDED_CLIENTDATA()

    @ATMT.condition(SSLv2_WAITING_CLIENTDATA, prio=2)
    def sslv2_no_more_ClientData(self):
        if False:
            print('Hello World!')
        raise self.SSLv2_CLOSE_NOTIFY()

    @ATMT.state()
    def SSLv2_ADDED_CLIENTDATA(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(SSLv2_ADDED_CLIENTDATA)
    def sslv2_should_send_ClientData(self):
        if False:
            return 10
        self.flush_records()
        raise self.SSLv2_SENT_CLIENTDATA()

    @ATMT.state()
    def SSLv2_SENT_CLIENTDATA(self):
        if False:
            print('Hello World!')
        raise self.SSLv2_WAITING_SERVERDATA()

    @ATMT.state()
    def SSLv2_WAITING_SERVERDATA(self):
        if False:
            return 10
        self.get_next_msg(0.3, 1)
        raise self.SSLv2_RECEIVED_SERVERDATA()

    @ATMT.state()
    def SSLv2_RECEIVED_SERVERDATA(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(SSLv2_RECEIVED_SERVERDATA)
    def sslv2_should_handle_ServerData(self):
        if False:
            i = 10
            return i + 15
        if not self.buffer_in:
            raise self.SSLv2_WAITING_CLIENTDATA()
        p = self.buffer_in[0]
        print('> Received: %r' % p.load)
        if p.load.startswith(b'goodbye'):
            raise self.SSLv2_CLOSE_NOTIFY()
        self.buffer_in = self.buffer_in[1:]
        raise self.SSLv2_HANDLED_SERVERDATA()

    @ATMT.state()
    def SSLv2_HANDLED_SERVERDATA(self):
        if False:
            while True:
                i = 10
        raise self.SSLv2_WAITING_CLIENTDATA()

    @ATMT.state()
    def SSLv2_CLOSE_NOTIFY(self):
        if False:
            print('Hello World!')
        "\n        There is no proper way to end an SSLv2 session.\n        We try and send a 'goodbye' message as a substitute.\n        "
        self.vprint()
        self.vprint("Trying to send a 'goodbye' to the server...")

    @ATMT.condition(SSLv2_CLOSE_NOTIFY)
    def sslv2_close_session(self):
        if False:
            while True:
                i = 10
        self.add_record()
        self.add_msg(Raw('goodbye'))
        try:
            self.flush_records()
        except Exception:
            self.vprint('Could not send our goodbye. The server probably stopped.')
        self.socket.close()
        raise self.FINAL()

    @ATMT.state()
    def TLS13_START(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.condition(TLS13_START)
    def tls13_should_add_ClientHello(self):
        if False:
            return 10
        supported_groups = ['secp256r1', 'secp384r1', 'x448']
        if conf.crypto_valid_advanced:
            supported_groups.append('x25519')
        self.add_record(is_tls13=False)
        if self.client_hello:
            p = self.client_hello
        else:
            if self.ciphersuite is None:
                c = 4865
            else:
                c = self.ciphersuite
            p = TLS13ClientHello(ciphers=c)
        ext = []
        ext += TLS_Ext_SupportedVersion_CH(versions=['TLS 1.3'])
        s = self.cur_session
        if s.tls13_psk_secret:
            if self.tls13_psk_mode == 'psk_dhe_ke':
                ext += TLS_Ext_PSKKeyExchangeModes(kxmodes='psk_dhe_ke')
                ext += TLS_Ext_SupportedGroups(groups=supported_groups)
                ext += TLS_Ext_KeyShare_CH(client_shares=[KeyShareEntry(group=self.curve)])
            else:
                ext += TLS_Ext_PSKKeyExchangeModes(kxmodes='psk_ke')
            if s.client_session_ticket:
                cs_cls = _tls_cipher_suites_cls[s.tls13_ticket_ciphersuite]
                hkdf = TLS13_HKDF(cs_cls.hash_alg.name.lower())
                hash_len = hkdf.hash.digest_size
                agems = int((time.time() - s.client_ticket_age) * 1000)
                obfuscated_age = agems + s.client_session_ticket_age_add & 4294967295
                psk_id = PSKIdentity(identity=s.client_session_ticket, obfuscated_ticket_age=obfuscated_age)
                psk_binder_entry = PSKBinderEntry(binder_len=hash_len, binder=b'\x00' * hash_len)
                ext += TLS_Ext_PreSharedKey_CH(identities=[psk_id], binders=[psk_binder_entry])
            else:
                hkdf = TLS13_HKDF('sha256')
                hash_len = hkdf.hash.digest_size
                psk_id = PSKIdentity(identity='Client_identity')
                psk_binder_entry = PSKBinderEntry(binder_len=hash_len, binder=b'\x00' * hash_len)
                ext += TLS_Ext_PreSharedKey_CH(identities=[psk_id], binders=[psk_binder_entry])
        else:
            ext += TLS_Ext_SupportedGroups(groups=supported_groups)
            ext += TLS_Ext_KeyShare_CH(client_shares=[KeyShareEntry(group=self.curve)])
            ext += TLS_Ext_SignatureAlgorithms(sig_algs=['sha256+rsaepss', 'sha256+rsa'])
        if self.server_name:
            ext += TLS_Ext_ServerName(servernames=[ServerName(servername=self.server_name)])
        p.ext = ext
        self.add_msg(p)
        raise self.TLS13_ADDED_CLIENTHELLO()

    @ATMT.state()
    def TLS13_ADDED_CLIENTHELLO(self):
        if False:
            for i in range(10):
                print('nop')
        raise self.TLS13_SENDING_CLIENTFLIGHT1()

    @ATMT.state()
    def TLS13_SENDING_CLIENTFLIGHT1(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(TLS13_SENDING_CLIENTFLIGHT1)
    def tls13_should_send_ClientFlight1(self):
        if False:
            return 10
        self.flush_records()
        raise self.TLS13_SENT_CLIENTFLIGHT1()

    @ATMT.state()
    def TLS13_SENT_CLIENTFLIGHT1(self):
        if False:
            return 10
        raise self.TLS13_WAITING_SERVERFLIGHT1()

    @ATMT.state()
    def TLS13_WAITING_SERVERFLIGHT1(self):
        if False:
            i = 10
            return i + 15
        self.get_next_msg()
        raise self.TLS13_RECEIVED_SERVERFLIGHT1()

    @ATMT.state()
    def TLS13_RECEIVED_SERVERFLIGHT1(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(TLS13_RECEIVED_SERVERFLIGHT1, prio=1)
    def tls13_should_handle_ServerHello(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        XXX We should check the ServerHello attributes for discrepancies with\n        our own ClientHello.\n        '
        self.raise_on_packet(TLS13ServerHello, self.TLS13_HANDLED_SERVERHELLO)

    @ATMT.condition(TLS13_RECEIVED_SERVERFLIGHT1, prio=2)
    def tls13_should_handle_HelloRetryRequest(self):
        if False:
            print('Hello World!')
        '\n        XXX We should check the ServerHello attributes for discrepancies with\n        our own ClientHello.\n        '
        self.raise_on_packet(TLS13HelloRetryRequest, self.TLS13_HELLO_RETRY_REQUESTED)

    @ATMT.condition(TLS13_RECEIVED_SERVERFLIGHT1, prio=3)
    def tls13_should_handle_AlertMessage_(self):
        if False:
            while True:
                i = 10
        self.raise_on_packet(TLSAlert, self.TLS13_HANDLED_ALERT_FROM_SERVERFLIGHT1)

    @ATMT.state()
    def TLS13_HANDLED_ALERT_FROM_SERVERFLIGHT1(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('Received Alert message !')
        self.vprint(self.cur_pkt.mysummary())
        raise self.CLOSE_NOTIFY()

    @ATMT.condition(TLS13_RECEIVED_SERVERFLIGHT1, prio=4)
    def tls13_missing_ServerHello(self):
        if False:
            while True:
                i = 10
        raise self.MISSING_SERVERHELLO()

    @ATMT.state()
    def TLS13_HELLO_RETRY_REQUESTED(self):
        if False:
            return 10
        pass

    @ATMT.condition(TLS13_HELLO_RETRY_REQUESTED)
    def tls13_should_add_ClientHello_Retry(self):
        if False:
            i = 10
            return i + 15
        s = self.cur_session
        s.tls13_retry = True
        self.add_record(is_tls13=False)
        hrr = s.handshake_messages_parsed[-1]
        if isinstance(hrr, TLS13HelloRetryRequest):
            pass
        ciphersuite = hrr.cipher
        if hrr.ext:
            for e in hrr.ext:
                if isinstance(e, TLS_Ext_KeyShare_HRR):
                    selected_group = e.selected_group
                if isinstance(e, TLS_Ext_SupportedVersion_SH):
                    selected_version = e.version
        if not selected_group or not selected_version:
            raise self.CLOSE_NOTIFY()
        ext = []
        ext += TLS_Ext_SupportedVersion_CH(versions=[_tls_version[selected_version]])
        if s.tls13_psk_secret:
            if self.tls13_psk_mode == 'psk_dhe_ke':
                ext += (TLS_Ext_PSKKeyExchangeModes(kxmodes='psk_dhe_ke'),)
                ext += TLS_Ext_SupportedGroups(groups=[_tls_named_groups[selected_group]])
                ext += TLS_Ext_KeyShare_CH(client_shares=[KeyShareEntry(group=selected_group)])
            else:
                ext += TLS_Ext_PSKKeyExchangeModes(kxmodes='psk_ke')
            if s.client_session_ticket:
                cs_cls = _tls_cipher_suites_cls[s.tls13_ticket_ciphersuite]
                hkdf = TLS13_HKDF(cs_cls.hash_alg.name.lower())
                hash_len = hkdf.hash.digest_size
                agems = int((time.time() - s.client_ticket_age) * 1000)
                obfuscated_age = agems + s.client_session_ticket_age_add & 4294967295
                psk_id = PSKIdentity(identity=s.client_session_ticket, obfuscated_ticket_age=obfuscated_age)
                psk_binder_entry = PSKBinderEntry(binder_len=hash_len, binder=b'\x00' * hash_len)
                ext += TLS_Ext_PreSharedKey_CH(identities=[psk_id], binders=[psk_binder_entry])
            else:
                hkdf = TLS13_HKDF('sha256')
                hash_len = hkdf.hash.digest_size
                psk_id = PSKIdentity(identity='Client_identity')
                psk_binder_entry = PSKBinderEntry(binder_len=hash_len, binder=b'\x00' * hash_len)
                ext += TLS_Ext_PreSharedKey_CH(identities=[psk_id], binders=[psk_binder_entry])
        else:
            ext += TLS_Ext_SupportedGroups(groups=[_tls_named_groups[selected_group]])
            ext += TLS_Ext_KeyShare_CH(client_shares=[KeyShareEntry(group=selected_group)])
            ext += TLS_Ext_SignatureAlgorithms(sig_algs=['sha256+rsaepss'])
        p = TLS13ClientHello(ciphers=ciphersuite, ext=ext)
        self.add_msg(p)
        raise self.TLS13_ADDED_CLIENTHELLO()

    @ATMT.state()
    def TLS13_HANDLED_SERVERHELLO(self):
        if False:
            while True:
                i = 10
        pass

    @ATMT.condition(TLS13_HANDLED_SERVERHELLO, prio=1)
    def tls13_should_handle_encrytpedExtensions(self):
        if False:
            return 10
        self.raise_on_packet(TLSEncryptedExtensions, self.TLS13_HANDLED_ENCRYPTEDEXTENSIONS)

    @ATMT.condition(TLS13_HANDLED_SERVERHELLO, prio=2)
    def tls13_should_handle_ChangeCipherSpec(self):
        if False:
            i = 10
            return i + 15
        self.raise_on_packet(TLSChangeCipherSpec, self.TLS13_HANDLED_CHANGE_CIPHER_SPEC)

    @ATMT.state()
    def TLS13_HANDLED_CHANGE_CIPHER_SPEC(self):
        if False:
            return 10
        self.cur_session.middlebox_compatibility = True
        raise self.TLS13_HANDLED_SERVERHELLO()

    @ATMT.condition(TLS13_HANDLED_SERVERHELLO, prio=3)
    def tls13_missing_encryptedExtension(self):
        if False:
            i = 10
            return i + 15
        self.vprint('Missing TLS 1.3 EncryptedExtensions message!')
        raise self.CLOSE_NOTIFY()

    @ATMT.state()
    def TLS13_HANDLED_ENCRYPTEDEXTENSIONS(self):
        if False:
            return 10
        pass

    @ATMT.condition(TLS13_HANDLED_ENCRYPTEDEXTENSIONS, prio=1)
    def tls13_should_handle_certificateRequest_from_encryptedExtensions(self):
        if False:
            return 10
        '\n        XXX We should check the CertificateRequest attributes for discrepancies\n        with the cipher suite, etc.\n        '
        self.raise_on_packet(TLS13CertificateRequest, self.TLS13_HANDLED_CERTIFICATEREQUEST)

    @ATMT.condition(TLS13_HANDLED_ENCRYPTEDEXTENSIONS, prio=2)
    def tls13_should_handle_certificate_from_encryptedExtensions(self):
        if False:
            i = 10
            return i + 15
        self.tls13_should_handle_Certificate()

    @ATMT.condition(TLS13_HANDLED_ENCRYPTEDEXTENSIONS, prio=3)
    def tls13_should_handle_finished_from_encryptedExtensions(self):
        if False:
            return 10
        if self.cur_session.tls13_psk_secret:
            self.raise_on_packet(TLSFinished, self.TLS13_HANDLED_FINISHED)

    @ATMT.state()
    def TLS13_HANDLED_CERTIFICATEREQUEST(self):
        if False:
            print('Hello World!')
        pass

    @ATMT.condition(TLS13_HANDLED_CERTIFICATEREQUEST, prio=1)
    def tls13_should_handle_Certificate_from_CertificateRequest(self):
        if False:
            return 10
        return self.tls13_should_handle_Certificate()

    def tls13_should_handle_Certificate(self):
        if False:
            i = 10
            return i + 15
        self.raise_on_packet(TLS13Certificate, self.TLS13_HANDLED_CERTIFICATE)

    @ATMT.state()
    def TLS13_HANDLED_CERTIFICATE(self):
        if False:
            i = 10
            return i + 15
        pass

    @ATMT.condition(TLS13_HANDLED_CERTIFICATE, prio=1)
    def tls13_should_handle_CertificateVerify(self):
        if False:
            print('Hello World!')
        self.raise_on_packet(TLSCertificateVerify, self.TLS13_HANDLED_CERTIFICATE_VERIFY)

    @ATMT.condition(TLS13_HANDLED_CERTIFICATE, prio=2)
    def tls13_missing_CertificateVerify(self):
        if False:
            return 10
        self.vprint('Missing TLS 1.3 CertificateVerify message!')
        raise self.CLOSE_NOTIFY()

    @ATMT.state()
    def TLS13_HANDLED_CERTIFICATE_VERIFY(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @ATMT.condition(TLS13_HANDLED_CERTIFICATE_VERIFY, prio=1)
    def tls13_should_handle_finished(self):
        if False:
            while True:
                i = 10
        self.raise_on_packet(TLSFinished, self.TLS13_HANDLED_FINISHED)

    @ATMT.state()
    def TLS13_HANDLED_FINISHED(self):
        if False:
            while True:
                i = 10
        raise self.TLS13_PREPARE_CLIENTFLIGHT2()

    @ATMT.state()
    def TLS13_PREPARE_CLIENTFLIGHT2(self):
        if False:
            while True:
                i = 10
        if self.cur_session.middlebox_compatibility:
            self.add_record(is_tls12=True)
            self.add_msg(TLSChangeCipherSpec())
        self.add_record(is_tls13=True)

    @ATMT.condition(TLS13_PREPARE_CLIENTFLIGHT2, prio=1)
    def tls13_should_add_ClientCertificate(self):
        if False:
            i = 10
            return i + 15
        '\n        If the server sent a CertificateRequest, we send a Certificate message.\n        If no certificate is available, an empty Certificate message is sent:\n        - this is a SHOULD in RFC 4346 (Section 7.4.6)\n        - this is a MUST in RFC 5246 (Section 7.4.6)\n\n        XXX We may want to add a complete chain.\n        '
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if TLS13CertificateRequest not in hs_msg:
            raise self.TLS13_ADDED_CLIENTCERTIFICATE()
        certs = []
        if self.mycert:
            certs += _ASN1CertAndExt(cert=self.mycert)
        self.add_msg(TLS13Certificate(certs=certs))
        raise self.TLS13_ADDED_CLIENTCERTIFICATE()

    @ATMT.state()
    def TLS13_ADDED_CLIENTCERTIFICATE(self):
        if False:
            return 10
        pass

    @ATMT.condition(TLS13_ADDED_CLIENTCERTIFICATE, prio=1)
    def tls13_should_add_ClientCertificateVerify(self):
        if False:
            print('Hello World!')
        '\n        XXX Section 7.4.7.1 of RFC 5246 states that the CertificateVerify\n        message is only sent following a client certificate that has signing\n        capability (i.e. not those containing fixed DH params).\n        We should verify that before adding the message. We should also handle\n        the case when the Certificate message was empty.\n        '
        hs_msg = [type(m) for m in self.cur_session.handshake_messages_parsed]
        if TLS13CertificateRequest not in hs_msg or self.mycert is None or self.mykey is None:
            return self.tls13_should_add_ClientFinished()
        self.add_msg(TLSCertificateVerify())
        raise self.TLS13_ADDED_CERTIFICATEVERIFY()

    @ATMT.state()
    def TLS13_ADDED_CERTIFICATEVERIFY(self):
        if False:
            return 10
        return self.tls13_should_add_ClientFinished()

    @ATMT.condition(TLS13_PREPARE_CLIENTFLIGHT2, prio=2)
    def tls13_should_add_ClientFinished(self):
        if False:
            print('Hello World!')
        self.add_msg(TLSFinished())
        raise self.TLS13_ADDED_CLIENTFINISHED()

    @ATMT.state()
    def TLS13_ADDED_CLIENTFINISHED(self):
        if False:
            return 10
        pass

    @ATMT.condition(TLS13_ADDED_CLIENTFINISHED)
    def tls13_should_send_ClientFlight2(self):
        if False:
            return 10
        self.flush_records()
        raise self.TLS13_SENT_CLIENTFLIGHT2()

    @ATMT.state()
    def TLS13_SENT_CLIENTFLIGHT2(self):
        if False:
            for i in range(10):
                print('nop')
        self.vprint('TLS 1.3 handshake completed!')
        self.vprint_sessioninfo()
        self.vprint("You may send data or use 'quit'.")
        raise self.WAIT_CLIENTDATA()

    @ATMT.state()
    def SOCKET_CLOSED(self):
        if False:
            i = 10
            return i + 15
        raise self.FINAL()

    @ATMT.state(stop=True)
    def STOP(self):
        if False:
            for i in range(10):
                print('nop')
        if self.cur_session.advertised_tls_version in [512, 2]:
            raise self.SSLv2_CLOSE_NOTIFY()
        else:
            raise self.CLOSE_NOTIFY()

    @ATMT.state(final=True)
    def FINAL(self):
        if False:
            print('Hello World!')
        self.vprint('Closing client socket...')
        self.socket.close()
        self.vprint('Ending TLS client automaton.')