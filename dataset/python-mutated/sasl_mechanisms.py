"""
Protocol agnostic implementations of SASL authentication mechanisms.
"""
import binascii
import os
import random
import time
from hashlib import md5
from zope.interface import Attribute, Interface, implementer
from twisted.python.compat import networkString

class ISASLMechanism(Interface):
    name = Attribute('Common name for the SASL Mechanism.')

    def getInitialResponse():
        if False:
            return 10
        '\n        Get the initial client response, if defined for this mechanism.\n\n        @return: initial client response string.\n        @rtype: C{str}.\n        '

    def getResponse(challenge):
        if False:
            while True:
                i = 10
        '\n        Get the response to a server challenge.\n\n        @param challenge: server challenge.\n        @type challenge: C{str}.\n        @return: client response.\n        @rtype: C{str}.\n        '

@implementer(ISASLMechanism)
class Anonymous:
    """
    Implements the ANONYMOUS SASL authentication mechanism.

    This mechanism is defined in RFC 2245.
    """
    name = 'ANONYMOUS'

    def getInitialResponse(self):
        if False:
            print('Hello World!')
        return None

    def getResponse(self, challenge):
        if False:
            for i in range(10):
                print('nop')
        pass

@implementer(ISASLMechanism)
class Plain:
    """
    Implements the PLAIN SASL authentication mechanism.

    The PLAIN SASL authentication mechanism is defined in RFC 2595.
    """
    name = 'PLAIN'

    def __init__(self, authzid, authcid, password):
        if False:
            print('Hello World!')
        '\n        @param authzid: The authorization identity.\n        @type authzid: L{unicode}\n\n        @param authcid: The authentication identity.\n        @type authcid: L{unicode}\n\n        @param password: The plain-text password.\n        @type password: L{unicode}\n        '
        self.authzid = authzid or ''
        self.authcid = authcid or ''
        self.password = password or ''

    def getInitialResponse(self):
        if False:
            return 10
        return self.authzid.encode('utf-8') + b'\x00' + self.authcid.encode('utf-8') + b'\x00' + self.password.encode('utf-8')

    def getResponse(self, challenge):
        if False:
            print('Hello World!')
        pass

@implementer(ISASLMechanism)
class DigestMD5:
    """
    Implements the DIGEST-MD5 SASL authentication mechanism.

    The DIGEST-MD5 SASL authentication mechanism is defined in RFC 2831.
    """
    name = 'DIGEST-MD5'

    def __init__(self, serv_type, host, serv_name, username, password):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param serv_type: An indication of what kind of server authentication\n            is being attempted against.  For example, C{u"xmpp"}.\n        @type serv_type: C{unicode}\n\n        @param host: The authentication hostname.  Also known as the realm.\n            This is used as a scope to help select the right credentials.\n        @type host: C{unicode}\n\n        @param serv_name: An additional identifier for the server.\n        @type serv_name: C{unicode}\n\n        @param username: The authentication username to use to respond to a\n            challenge.\n        @type username: C{unicode}\n\n        @param password: The authentication password to use to respond to a\n            challenge.\n        @type password: C{unicode}\n        '
        self.username = username
        self.password = password
        self.defaultRealm = host
        self.digest_uri = f'{serv_type}/{host}'
        if serv_name is not None:
            self.digest_uri += f'/{serv_name}'

    def getInitialResponse(self):
        if False:
            print('Hello World!')
        return None

    def getResponse(self, challenge):
        if False:
            return 10
        directives = self._parse(challenge)
        if b'rspauth' in directives:
            return b''
        charset = directives[b'charset'].decode('ascii')
        try:
            realm = directives[b'realm']
        except KeyError:
            realm = self.defaultRealm.encode(charset)
        return self._genResponse(charset, realm, directives[b'nonce'])

    def _parse(self, challenge):
        if False:
            return 10
        '\n        Parses the server challenge.\n\n        Splits the challenge into a dictionary of directives with values.\n\n        @return: challenge directives and their values.\n        @rtype: C{dict} of C{str} to C{str}.\n        '
        s = challenge
        paramDict = {}
        cur = 0
        remainingParams = True
        while remainingParams:
            middle = s.index(b'=', cur)
            name = s[cur:middle].lstrip()
            middle += 1
            if s[middle:middle + 1] == b'"':
                middle += 1
                end = s.index(b'"', middle)
                value = s[middle:end]
                cur = s.find(b',', end) + 1
                if cur == 0:
                    remainingParams = False
            else:
                end = s.find(b',', middle)
                if end == -1:
                    value = s[middle:].rstrip()
                    remainingParams = False
                else:
                    value = s[middle:end].rstrip()
                cur = end + 1
            paramDict[name] = value
        for param in (b'qop', b'cipher'):
            if param in paramDict:
                paramDict[param] = paramDict[param].split(b',')
        return paramDict

    def _unparse(self, directives):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create message string from directives.\n\n        @param directives: dictionary of directives (names to their values).\n                           For certain directives, extra quotes are added, as\n                           needed.\n        @type directives: C{dict} of C{str} to C{str}\n        @return: message string.\n        @rtype: C{str}.\n        '
        directive_list = []
        for (name, value) in directives.items():
            if name in (b'username', b'realm', b'cnonce', b'nonce', b'digest-uri', b'authzid', b'cipher'):
                directive = name + b'=' + value
            else:
                directive = name + b'=' + value
            directive_list.append(directive)
        return b','.join(directive_list)

    def _calculateResponse(self, cnonce, nc, nonce, username, password, realm, uri):
        if False:
            while True:
                i = 10
        '\n        Calculates response with given encoded parameters.\n\n        @return: The I{response} field of a response to a Digest-MD5 challenge\n            of the given parameters.\n        @rtype: L{bytes}\n        '

        def H(s):
            if False:
                i = 10
                return i + 15
            return md5(s).digest()

        def HEX(n):
            if False:
                i = 10
                return i + 15
            return binascii.b2a_hex(n)

        def KD(k, s):
            if False:
                while True:
                    i = 10
            return H(k + b':' + s)
        a1 = H(username + b':' + realm + b':' + password) + b':' + nonce + b':' + cnonce
        a2 = b'AUTHENTICATE:' + uri
        response = HEX(KD(HEX(H(a1)), nonce + b':' + nc + b':' + cnonce + b':' + b'auth' + b':' + HEX(H(a2))))
        return response

    def _genResponse(self, charset, realm, nonce):
        if False:
            print('Hello World!')
        '\n        Generate response-value.\n\n        Creates a response to a challenge according to section 2.1.2.1 of\n        RFC 2831 using the C{charset}, C{realm} and C{nonce} directives\n        from the challenge.\n        '
        try:
            username = self.username.encode(charset)
            password = self.password.encode(charset)
            digest_uri = self.digest_uri.encode(charset)
        except UnicodeError:
            raise
        nc = networkString(f'{1:08x}')
        cnonce = self._gen_nonce()
        qop = b'auth'
        response = self._calculateResponse(cnonce, nc, nonce, username, password, realm, digest_uri)
        directives = {b'username': username, b'realm': realm, b'nonce': nonce, b'cnonce': cnonce, b'nc': nc, b'qop': qop, b'digest-uri': digest_uri, b'response': response, b'charset': charset.encode('ascii')}
        return self._unparse(directives)

    def _gen_nonce(self):
        if False:
            print('Hello World!')
        nonceString = '%f:%f:%d' % (random.random(), time.time(), os.getpid())
        nonceBytes = networkString(nonceString)
        return md5(nonceBytes).hexdigest().encode('ascii')