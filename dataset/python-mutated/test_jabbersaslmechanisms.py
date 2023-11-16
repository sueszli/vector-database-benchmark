"""
Tests for L{twisted.words.protocols.jabber.sasl_mechanisms}.
"""
from twisted.python.compat import networkString
from twisted.trial import unittest
from twisted.words.protocols.jabber import sasl_mechanisms

class PlainTests(unittest.TestCase):
    """
    Tests for L{twisted.words.protocols.jabber.sasl_mechanisms.Plain}.
    """

    def test_getInitialResponse(self) -> None:
        if False:
            return 10
        '\n        Test the initial response.\n        '
        m = sasl_mechanisms.Plain(None, 'test', 'secret')
        self.assertEqual(m.getInitialResponse(), b'\x00test\x00secret')

class AnonymousTests(unittest.TestCase):
    """
    Tests for L{twisted.words.protocols.jabber.sasl_mechanisms.Anonymous}.
    """

    def test_getInitialResponse(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the initial response to be empty.\n        '
        m = sasl_mechanisms.Anonymous()
        self.assertEqual(m.getInitialResponse(), None)

class DigestMD5Tests(unittest.TestCase):
    """
    Tests for L{twisted.words.protocols.jabber.sasl_mechanisms.DigestMD5}.
    """

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.mechanism = sasl_mechanisms.DigestMD5('xmpp', 'example.org', None, 'test', 'secret')

    def test_getInitialResponse(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that no initial response is generated.\n        '
        self.assertIdentical(self.mechanism.getInitialResponse(), None)

    def test_getResponse(self) -> None:
        if False:
            return 10
        '\n        The response to a Digest-MD5 challenge includes the parameters from the\n        challenge.\n        '
        challenge = b'realm="localhost",nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
        directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
        del directives[b'cnonce'], directives[b'response']
        self.assertEqual({b'username': b'test', b'nonce': b'1234', b'nc': b'00000001', b'qop': [b'auth'], b'charset': b'utf-8', b'realm': b'localhost', b'digest-uri': b'xmpp/example.org'}, directives)

    def test_getResponseNonAsciiRealm(self) -> None:
        if False:
            return 10
        '\n        Bytes outside the ASCII range in the challenge are nevertheless\n        included in the response.\n        '
        challenge = b'realm="\xc3\xa9chec.example.org",nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
        directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
        del directives[b'cnonce'], directives[b'response']
        self.assertEqual({b'username': b'test', b'nonce': b'1234', b'nc': b'00000001', b'qop': [b'auth'], b'charset': b'utf-8', b'realm': b'\xc3\xa9chec.example.org', b'digest-uri': b'xmpp/example.org'}, directives)

    def test_getResponseNoRealm(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The response to a challenge without a realm uses the host part of the\n        JID as the realm.\n        '
        challenge = b'nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
        directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
        self.assertEqual(directives[b'realm'], b'example.org')

    def test_getResponseNoRealmIDN(self) -> None:
        if False:
            return 10
        '\n        If the challenge does not include a realm and the host part of the JID\n        includes bytes outside of the ASCII range, the response still includes\n        the host part of the JID as the realm.\n        '
        self.mechanism = sasl_mechanisms.DigestMD5('xmpp', 'échec.example.org', None, 'test', 'secret')
        challenge = b'nonce="1234",qop="auth",charset=utf-8,algorithm=md5-sess'
        directives = self.mechanism._parse(self.mechanism.getResponse(challenge))
        self.assertEqual(directives[b'realm'], b'\xc3\xa9chec.example.org')

    def test_getResponseRspauth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        If the challenge just has a rspauth directive, the response is empty.\n        '
        challenge = b'rspauth=cnNwYXV0aD1lYTQwZjYwMzM1YzQyN2I1NTI3Yjg0ZGJhYmNkZmZmZA=='
        response = self.mechanism.getResponse(challenge)
        self.assertEqual(b'', response)

    def test_calculateResponse(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The response to a Digest-MD5 challenge is computed according to RFC\n        2831.\n        '
        charset = 'utf-8'
        nonce = b'OA6MG9tEQGm2hh'
        nc = networkString(f'{1:08x}')
        cnonce = b'OA6MHXh6VqTrRk'
        username = 'Иchris'
        password = 'Иsecret'
        host = 'Иelwood.innosoft.com'
        digestURI = 'imap/Иelwood.innosoft.com'.encode(charset)
        mechanism = sasl_mechanisms.DigestMD5(b'imap', host, None, username, password)
        response = mechanism._calculateResponse(cnonce, nc, nonce, username.encode(charset), password.encode(charset), host.encode(charset), digestURI)
        self.assertEqual(response, b'7928f233258be88392424d094453c5e3')

    def test_parse(self) -> None:
        if False:
            print('Hello World!')
        '\n        A challenge can be parsed into a L{dict} with L{bytes} or L{list}\n        values.\n        '
        challenge = b'nonce="1234",qop="auth,auth-conf",charset=utf-8,algorithm=md5-sess,cipher="des,3des"'
        directives = self.mechanism._parse(challenge)
        self.assertEqual({b'algorithm': b'md5-sess', b'nonce': b'1234', b'charset': b'utf-8', b'qop': [b'auth', b'auth-conf'], b'cipher': [b'des', b'3des']}, directives)